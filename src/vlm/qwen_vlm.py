from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


class QwenVLMExtractor:
    def __init__(self, model_name: str, device: str, prompt_path: str | Path = "", prompt_text: str = "", logger=None):
        self.model_name = model_name
        self.device = device
        self.logger = logger
        if prompt_text:
            self.prompt = prompt_text
        elif prompt_path:
            self.prompt = Path(prompt_path).read_text(encoding="utf-8")
        else:
            self.prompt = ""
        try:
            import torch
            from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration

            self.torch = torch
            dtype = torch.float16 if device == "cuda" else torch.float32
            self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
            )
            if device != "cuda":
                self.model.to(device)
            self._error = None
        except ImportError:
            try:
                import torch
                from transformers import AutoProcessor, AutoModelForVision2Seq

                self.torch = torch
                dtype = torch.float16 if device == "cuda" else torch.float32
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                )
                if device != "cuda":
                    self.model.to(device)
                self._error = None
            except Exception as exc2:
                self._error = str(exc2)
        except Exception as exc:
            self._error = str(exc)

    def is_available(self) -> bool:
        return not self._error

    def extract(self, image_path: str) -> dict:
        if not self.is_available():
            return {"error": f"Qwen VLM unavailable: {self._error}"}
        try:
            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            ).to(self.device)

            with self.torch.no_grad():
                ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids = ids[0][inputs["input_ids"].shape[-1]:]
            out = self.processor.decode(generated_ids, skip_special_tokens=True)

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
            return {"error": "Model did not return JSON", "raw_output": out[:2000]}
        except Exception as exc:
            return {"error": str(exc)}