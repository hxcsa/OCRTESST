from __future__ import annotations

import json


class GemmaTextExtractor:
    def __init__(self, model_name: str, device: str, logger=None):
        self.model_name = model_name
        self.device = device
        self.logger = logger
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.torch = torch
            dtype = torch.float16 if device == "cuda" else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
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

    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        if not self.is_available():
            raise RuntimeError(f"Gemma model unavailable: {self._error}")
        chat = [
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2,
            )
        generated_ids = ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)