from __future__ import annotations

import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = """You are an OCR text cleaner for scanned Kurdish (Sorani) and Arabic government documents.
Fix all OCR errors: correct garbled characters, rejoin split words, fix reversed/misread Arabic digits, and restore proper formatting.
Preserve the original meaning and structure. Output ONLY the cleaned text — no explanations, no markdown, no extra text.

OCR text to clean:
{text}

Cleaned text:"""

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


class LLMCleanupEngine:
    def __init__(self, device: str = "cuda", logger=None):
        self.device = device
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._loaded = True

    def unload(self):
        if self.model:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded = False
            gc.collect()
            torch.cuda.empty_cache()

    def clean(self, text: str, max_new_tokens: int = 1024) -> tuple[str, float]:
        if not text or not text.strip():
            return text, 0.0
        self.load()
        prompt = PROMPT_TEMPLATE.format(text=text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0

        generated = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
        return generated, elapsed

    def clean_file(self, txt_path: str | Path, output_path: str | Path | None = None) -> str:
        txt_path = Path(txt_path)
        raw = txt_path.read_text()
        cleaned, elapsed = self.clean(raw)
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(cleaned)
        if self.logger:
            self.logger.info("LLM cleanup of %s done in %.1fs", txt_path.name, elapsed)
        return cleaned
