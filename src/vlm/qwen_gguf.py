from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from src.benchmark import preprocess
from src.io_utils import load_manifest, safe_write_json
from src.runtime import measured_run
from src.extraction.validators import validate_extraction
from src.training.kurdish_normalization import normalize_json_strings


PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "qwen_gguf_document_prompt.txt"
CONTENT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "qwen_gguf_content_prompt.txt"

EXTRACTION_KEYS = [
    "recipient",
    "subject",
    "document_number",
    "document_date",
    "project_name",
    "location",
    "area_m2",
    "referenced_decision_number",
    "attachments",
    "sender_or_department",
    "full_name",
    "national_id",
    "phone_number",
    "address",
    "decision_or_status",
    "stamp_present",
    "signature_present",
    "uncertain_lines",
    "extraction_confidence",
]


def _data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        payload = text[start : end + 1]
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", payload)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return {"error": f"JSON parse failed: {exc}", "raw_output": text[:4000]}
    return {"error": "Model did not return a JSON object", "raw_output": text[:4000]}


def _clean_document_number(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if re.search(r"[\d\u0660-\u0669\u06f0-\u06f9]", value) and re.search(r"[A-Za-z]", value):
        cleaned = re.sub(r"[A-Za-z]+", "", value)
        cleaned = re.sub(r"\s+", "", cleaned).strip(":-/ ")
        return cleaned or value
    return value


def _clean_schema(data: dict[str, Any], *, normalize_output_text: bool = True) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key in EXTRACTION_KEYS:
        if key in ("attachments", "uncertain_lines"):
            value = data.get(key, [])
            cleaned[key] = value if isinstance(value, list) else [str(value)]
        elif key in ("stamp_present", "signature_present"):
            value = data.get(key)
            cleaned[key] = value if isinstance(value, bool) else bool(value) if value is not None else False
        elif key == "extraction_confidence":
            value = data.get(key, 0.0)
            try:
                cleaned[key] = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                cleaned[key] = 0.0
        else:
            value = data.get(key)
            if isinstance(value, list):
                value = "; ".join(str(item) for item in value[:3])
            if key in ("document_number", "referenced_decision_number"):
                value = _clean_document_number(value)
            cleaned[key] = value if value not in ("", "null", "None") else None
    if data.get("error"):
        cleaned["error"] = data["error"]
    if data.get("raw_output"):
        cleaned["raw_output"] = data["raw_output"]
    if normalize_output_text:
        cleaned = normalize_json_strings(cleaned)
    return cleaned


def download_qwen_gguf(cfg: dict[str, Any], logger=None) -> tuple[Path, Path]:
    qcfg = cfg.get("qwen_gguf", {})
    repo_id = qcfg.get("repo_id", "jc-builds/Qwen3.5-9B-VLM-Q4_K_M-GGUF")
    local_dir = Path(qcfg.get("local_dir", "models/Qwen3.5-9B-VLM-Q4_K_M-GGUF"))
    model_filename = qcfg.get("model_filename", "Qwen3.5-9B-Q4_K_M.gguf")
    mmproj_filename = qcfg.get("mmproj_filename", "mmproj-F16.gguf")
    local_dir.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info("Downloading/checking Qwen GGUF files from %s", repo_id)
    model_path = Path(hf_hub_download(repo_id=repo_id, filename=model_filename, local_dir=local_dir))
    mmproj_path = Path(hf_hub_download(repo_id=repo_id, filename=mmproj_filename, local_dir=local_dir))
    return model_path, mmproj_path


class QwenGGUFExtractor:
    def __init__(self, model_path: str | Path, mmproj_path: str | Path, cfg: dict[str, Any], logger=None):
        self.logger = logger
        qcfg = cfg.get("qwen_gguf", {})
        self.prompt = PROMPT_PATH.read_text(encoding="utf-8")
        self.model_path = Path(model_path)
        self.mmproj_path = Path(mmproj_path)
        self.llama_cli = Path(qcfg.get("llama_cli", "tools/llama.cpp-cuda/llama-cli"))
        if not self.llama_cli.is_absolute():
            self.llama_cli = Path.cwd() / self.llama_cli
        if not self.llama_cli.exists():
            raise FileNotFoundError(f"llama-cli not found: {self.llama_cli}")
        self.n_ctx = int(qcfg.get("n_ctx", 8192))
        self.n_gpu_layers = int(qcfg.get("n_gpu_layers", -1))
        self.max_tokens = int(qcfg.get("max_tokens", 1400))
        self.temperature = float(qcfg.get("temperature", 0.0))
        self.top_p = float(qcfg.get("top_p", 0.1))
        self.top_k = int(qcfg.get("top_k", 1))
        self.min_p = float(qcfg.get("min_p", 0.0))
        self.seed = int(qcfg.get("seed", 0))
        self.repeat_penalty = float(qcfg.get("repeat_penalty", 1.18))

    def extract(self, image_path: str | Path) -> dict[str, Any]:
        env = os.environ.copy()
        lib_dir = str(self.llama_cli.parent)
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        cmd = [
            str(self.llama_cli),
            "--model",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "--image",
            str(image_path),
            "--prompt",
            self.prompt,
            "--ctx-size",
            str(self.n_ctx),
            "--gpu-layers",
            str(self.n_gpu_layers),
            "--mmproj-offload",
            "--predict",
            str(self.max_tokens),
            "--temp",
            str(self.temperature),
            "--top-p",
            str(self.top_p),
            "--top-k",
            str(self.top_k),
            "--min-p",
            str(self.min_p),
            "--seed",
            str(self.seed),
            "--no-display-prompt",
            "--simple-io",
        ]
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            return {"error": f"llama-cli failed with exit code {proc.returncode}", "stderr": proc.stderr[-4000:]}
        data = _parse_json(proc.stdout)
        if self.logger and proc.stderr:
            self.logger.info("llama-cli stderr tail: %s", proc.stderr[-1200:])
        return data


class QwenGGUFServerExtractor:
    def __init__(self, model_path: str | Path, mmproj_path: str | Path, cfg: dict[str, Any], logger=None, prompt_path: str | Path = PROMPT_PATH):
        self.logger = logger
        qcfg = cfg.get("qwen_gguf", {})
        self.prompt = Path(prompt_path).read_text(encoding="utf-8")
        self.model_path = Path(model_path)
        self.mmproj_path = Path(mmproj_path)
        self.server = Path(qcfg.get("llama_server", "tools/llama.cpp-cuda/llama-server"))
        if not self.server.is_absolute():
            self.server = Path.cwd() / self.server
        if not self.server.exists():
            raise FileNotFoundError(f"llama-server not found: {self.server}")
        self.host = qcfg.get("server_host", "127.0.0.1")
        self.port = int(qcfg.get("server_port", 8088))
        self.n_ctx = int(qcfg.get("n_ctx", 8192))
        self.n_gpu_layers = int(qcfg.get("n_gpu_layers", -1))
        self.image_max_tokens = int(qcfg.get("image_max_tokens", 1024))
        self.max_tokens = int(qcfg.get("max_tokens", 1400))
        self.temperature = float(qcfg.get("temperature", 0.0))
        self.top_p = float(qcfg.get("top_p", 0.1))
        self.top_k = int(qcfg.get("top_k", 1))
        self.min_p = float(qcfg.get("min_p", 0.0))
        self.seed = int(qcfg.get("seed", 0))
        self.repeat_penalty = float(qcfg.get("repeat_penalty", 1.18))
        self.log_path = Path("reports") / "qwen_gguf_llama_server.log"
        self.log_file = None
        self.process: subprocess.Popen[str] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        env = os.environ.copy()
        lib_dir = str(self.server.parent)
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        cmd = [
            str(self.server),
            "--model",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.n_ctx),
            "--gpu-layers",
            str(self.n_gpu_layers),
            "--mmproj-offload",
            "--image-max-tokens",
            str(self.image_max_tokens),
            "--parallel",
            "1",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
        ]
        if self.logger:
            self.logger.info("Starting CUDA llama-server on %s", self.base_url)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(cmd, env=env, text=True, stdout=self.log_file, stderr=subprocess.STDOUT)
        self._wait_until_ready()

    def stop(self) -> None:
        if not self.process:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def _wait_until_ready(self) -> None:
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if self.log_path.exists() else ""
                raise RuntimeError(f"llama-server exited early:\n{tail}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            time.sleep(1)
        tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if self.log_path.exists() else ""
        raise TimeoutError(f"llama-server did not become ready. Last output:\n{tail}")

    def complete(self, image_path: str | Path, prompt: str | None = None, response_format_json: bool = False) -> str:
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or self.prompt},
                        {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "seed": self.seed,
            "repeat_penalty": self.repeat_penalty,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return json.dumps({"error": f"llama-server HTTP {exc.code}", "raw_output": exc.read().decode("utf-8", errors="replace")[:4000]})
        return body["choices"][0]["message"]["content"]

    def extract(self, image_path: str | Path, transcript: str | None = None) -> dict[str, Any]:
        prompt = self.prompt
        if transcript:
            prompt = (
                f"{self.prompt}\n\n"
                "Use this separately generated page transcription as a linguistic anchor. "
                "Prefer values that are visible in both the image and this text. "
                "If the transcription and image conflict, trust the image and record ambiguity in uncertain_lines.\n\n"
                f"PAGE_TRANSCRIPTION_MARKDOWN:\n{transcript[:3000]}"
            )
        text = self.complete(image_path, prompt, response_format_json=True)
        return _parse_json(text)

    def transcribe(self, image_path: str | Path) -> str:
        return self.complete(image_path, self.prompt).strip()


def _transcript_path(cfg: dict[str, Any], page: dict[str, Any]) -> Path:
    return Path(cfg["output_dir"]) / "transcriptions" / "qwen_gguf" / f"{page['document_id']}_p{int(page['page']):03d}_qwen_gguf.md"


def _load_transcript(cfg: dict[str, Any], page: dict[str, Any]) -> str | None:
    path = _transcript_path(cfg, page)
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        return text or None
    return None


def run_qwen_gguf(cfg: dict[str, Any], logger=None, limit: int | None = None, overwrite: bool = False) -> None:
    manifest = load_manifest(cfg["processed_dir"])
    pages = manifest.get("pages", []) if isinstance(manifest, dict) else manifest
    if not pages:
        pages = preprocess(cfg, logger)

    qcfg = cfg.get("qwen_gguf", {})
    image_variant = qcfg.get("image_variant", "raw")
    normalize_output_text = bool(qcfg.get("normalize_output_text", True))
    use_transcription_anchor = bool(qcfg.get("use_transcription_anchor", True))
    if limit is None:
        limit = int(qcfg.get("max_images", 5))
    selected = pages if limit <= 0 else pages[:limit]
    run_started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"qwen_gguf_{run_started_at}"
    model_path, mmproj_path = download_qwen_gguf(cfg, logger)
    extractor = QwenGGUFServerExtractor(model_path, mmproj_path, cfg, logger)
    out_dir = Path(cfg["output_dir"]) / "structured_json" / "vlm_qwen_gguf"
    run_out_dir = out_dir / "runs" / run_id

    try:
        extractor.start()
        for page in selected:
            image_path = page.get("images", {}).get(image_variant) or page["raw_image"]
            if logger:
                logger.info("Qwen GGUF extracting %s page %s image=%s", page["document_id"], page["page"], image_path)
            transcript = _load_transcript(cfg, page) if use_transcription_anchor else None
            with measured_run():
                data = _clean_schema(extractor.extract(image_path, transcript=transcript), normalize_output_text=normalize_output_text)
                if data.get("error") and transcript:
                    if logger:
                        logger.warning("Anchored extraction failed for %s page %s; retrying image-only", page["document_id"], page["page"])
                    data = _clean_schema(extractor.extract(image_path, transcript=None), normalize_output_text=normalize_output_text)
                data.update(validate_extraction(data))
            data.update(
                {
                    "document_id": page["document_id"],
                    "page": int(page["page"]),
                    "engine": "vlm_qwen_gguf",
                    "preprocessing_mode": image_variant,
                    "source": page.get("source"),
                    "image_path": image_path,
                    "run_id": run_id,
                    "extracted_at": run_started_at,
                    "runtime_seconds": measured_run.last.runtime_seconds,
                    "gpu_memory_mb": measured_run.last.gpu_memory_mb,
                    "transcription_used": bool(transcript),
                }
            )
            filename = f"{page['document_id']}_p{int(page['page']):03d}_vlm_qwen_gguf.json"
            safe_write_json(out_dir / filename, data, overwrite=overwrite)
            safe_write_json(run_out_dir / filename, data, overwrite=True)
    finally:
        extractor.stop()


def run_qwen_gguf_first_five(cfg: dict[str, Any], logger=None) -> None:
    run_qwen_gguf(cfg, logger=logger, limit=5, overwrite=False)


def run_qwen_gguf_transcription(cfg: dict[str, Any], logger=None, limit: int | None = None, overwrite: bool = False) -> None:
    manifest = load_manifest(cfg["processed_dir"])
    pages = manifest.get("pages", []) if isinstance(manifest, dict) else manifest
    if not pages:
        pages = preprocess(cfg, logger)

    qcfg = cfg.get("qwen_gguf", {})
    image_variant = qcfg.get("image_variant", "raw")
    if limit is None:
        limit = int(qcfg.get("max_images", 5))
    selected = pages if limit <= 0 else pages[:limit]
    run_started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"qwen_gguf_transcription_{run_started_at}"
    model_path, mmproj_path = download_qwen_gguf(cfg, logger)
    extractor = QwenGGUFServerExtractor(model_path, mmproj_path, cfg, logger, prompt_path=CONTENT_PROMPT_PATH)
    flat_dir = Path(cfg["output_dir"]) / "transcriptions" / "qwen_gguf"
    run_dir = flat_dir / "runs" / run_id

    try:
        extractor.start()
        for page in selected:
            image_path = page.get("images", {}).get(image_variant) or page["raw_image"]
            if logger:
                logger.info("Qwen GGUF transcribing %s page %s image=%s", page["document_id"], page["page"], image_path)
            with measured_run():
                text = extractor.transcribe(image_path)
            header = (
                f"<!-- run_id: {run_id} -->\n"
                f"<!-- extracted_at: {run_started_at} -->\n"
                f"<!-- document_id: {page['document_id']} page: {int(page['page'])} -->\n\n"
            )
            filename = f"{page['document_id']}_p{int(page['page']):03d}_qwen_gguf.md"
            out_text = header + text.strip() + "\n"
            flat_path = flat_dir / filename
            run_path = run_dir / filename
            flat_path.parent.mkdir(parents=True, exist_ok=True)
            run_path.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not flat_path.exists():
                flat_path.write_text(out_text, encoding="utf-8")
            run_path.write_text(out_text, encoding="utf-8")
    finally:
        extractor.stop()
