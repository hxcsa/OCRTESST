from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


class QwenVLMExtractor:
    def __init__(self, model_name: str, device: str, prompt_path: str | Path, logger=None):
        self.model_name = model_name
        self.device = device
        self.prompt = Path(prompt_path).read_text(encoding="utf-8")
        self.logger = logger
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            self.torch = torch
            dtype = torch.float16 if device == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
            )
            if device != "cuda":
                self.model.to(device)
            self._error = None
        except Exception as exc:
            self._error = str(exc)

    def is_available(self) -> bool:
        return not self._error

    def extract(self, image_path: str) -> dict:
        if not self.is_available():
            return {"error": f"Qwen VLM unavailable: {self._error}"}
        try:
            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": self.prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
            ids = self.model.generate(**inputs, max_new_tokens=1024)
            out = self.processor.batch_decode(ids, skip_special_tokens=True)[0]

            # The model echoes the full conversation; isolate the assistant response.
            # Use the last occurrence because "assistant" may appear earlier
            # (e.g. "You are a helpful assistant" in the system prompt).
            assistant_marker = "\nassistant\n"
            if assistant_marker in out:
                out = out.split(assistant_marker, 1)[-1]
            elif "assistant" in out:
                out = out.rsplit("assistant", 1)[-1]

            # Strip markdown code fences if present.
            out = out.strip()
            if out.startswith("```json"):
                out = out[7:]
            elif out.startswith("```"):
                out = out[3:]
            if out.endswith("```"):
                out = out[:-3]
            out = out.strip()

            start, end = out.find("{"), out.rfind("}")
            if start >= 0 and end > start:
                return json.loads(out[start : end + 1])
            return {"error": "Model did not return JSON", "raw_output": out}
        except Exception as exc:
            return {"error": str(exc)}
