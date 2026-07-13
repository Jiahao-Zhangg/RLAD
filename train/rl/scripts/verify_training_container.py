#!/usr/bin/env python3
"""Fail fast unless the imported training image satisfies RLAD's runtime contract."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path


def main() -> None:
    expected_megatron = Path("/root/Megatron-LM")
    expected_miles = Path(os.environ["MILES_DIR"]).resolve()
    if not expected_megatron.is_dir():
        raise SystemExit(f"missing Megatron-LM checkout: {expected_megatron}")
    if not expected_miles.is_dir():
        raise SystemExit(f"mounted miles checkout is missing: {expected_miles}")

    loaded = {
        name: importlib.import_module(name)
        for name in (
            "apex",
            "flash_attn",
            "megatron",
            "miles",
            "ray",
            "sglang",
            "torch",
            "transformer_engine",
        )
    }
    torch = loaded["torch"]
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot see the allocated GPU inside the container")
    if not torch.distributed.is_nccl_available():
        raise SystemExit("the container's PyTorch build does not provide NCCL")

    miles_file = Path(loaded["miles"].__file__).resolve()
    if expected_miles != miles_file and expected_miles not in miles_file.parents:
        raise SystemExit(
            f"imported miles from {miles_file}, expected the pinned host checkout {expected_miles}"
        )

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 9:
        raise SystemExit(f"expected an H100-class GPU, got compute capability {capability}")

    print(json.dumps({
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "device_capability": capability,
        "megatron_root": str(expected_megatron),
        "miles": str(miles_file),
        "torch": torch.__version__,
    }, indent=2))


if __name__ == "__main__":
    main()
