from __future__ import annotations

import argparse
import json


KURDISH_PROBE = "ڕ ڵ چ گ ک ی ە | ڕێباز هیوا | پڕۆژەی داون تاون - سلێمانی"


def check_tokenizer(model_name: str, text: str = KURDISH_PROBE) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    return {
        "model_name": model_name,
        "input": text,
        "decoded": decoded,
        "roundtrip_ok": decoded == text,
        "token_count": len(ids),
        "tokens": tokenizer.convert_ids_to_tokens(ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--text", default=KURDISH_PROBE)
    args = parser.parse_args()
    print(json.dumps(check_tokenizer(args.model, args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
