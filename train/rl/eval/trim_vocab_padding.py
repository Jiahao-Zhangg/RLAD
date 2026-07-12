"""Trim Megatron vocab padding from an exported HF checkpoint (idempotent).

miles' --save-hf export keeps Megatron's TP-divisibility padding on
embed_tokens/lm_head (e.g. 152,064 = 151,936 + 128 under TP2), while
config.json keeps the true vocab_size — vLLM asserts on the mismatch
(G2 probe 4767578). Padding rows sit at the tail and are never trained.

Usage: python trim_vocab_padding.py <hf_dir>
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sanitize_tokenizer_config(d: Path) -> None:
    """miles/Megatron's auto_bridge export writes `extra_special_tokens` as a LIST of
    token strings, but transformers expects a {attr_name: token} DICT and crashes in
    AutoTokenizer.from_pretrained (`'list' object has no attribute 'keys'`). These tokens
    are already declared in tokenizer.json/added_tokens_decoder, so dropping the bad field
    restores the base-tokenizer behavior. Idempotent."""
    tc = d / "tokenizer_config.json"
    if not tc.exists():
        return
    cfg = json.loads(tc.read_text())
    extra = cfg.get("extra_special_tokens")
    if isinstance(extra, list):
        cfg.pop("extra_special_tokens")
        tc.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
        print(f"{tc.name}: dropped malformed list-typed extra_special_tokens ({len(extra)} toks)")


def main(hf_dir: str) -> None:
    d = Path(hf_dir)
    sanitize_tokenizer_config(d)
    cfg = json.loads((d / "config.json").read_text())
    vocab = cfg["vocab_size"]
    st_paths = sorted(d.glob("model*.safetensors"))
    if not st_paths:
        print(f"no safetensors files under {d} — nothing to trim")
        return
    for st_path in st_paths:
        tensors = load_file(str(st_path))
        changed = False
        for key in list(tensors):
            if key in ("model.embed_tokens.weight", "lm_head.weight"):
                t = tensors[key]
                if t.shape[0] > vocab:
                    print(f"{st_path.name}:{key}: {tuple(t.shape)} -> ({vocab}, {t.shape[1]})")
                    tensors[key] = t[:vocab].contiguous()
                    changed = True
                elif t.shape[0] == vocab:
                    print(f"{st_path.name}:{key}: already {tuple(t.shape)} (no-op)")
                else:
                    sys.exit(f"FATAL: {key} rows {t.shape[0]} < vocab_size {vocab}")
        if changed:
            save_file(tensors, str(st_path), metadata={"format": "pt"})
            print(f"rewrote {st_path}")


if __name__ == "__main__":
    main(sys.argv[1])
