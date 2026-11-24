"""Merge a local PEFT adapter into a locally staged base for GGUF conversion."""

import argparse
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(base: Path, adapter: Path, target: Path):
    target.mkdir(parents=True, exist_ok=False)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, use_safetensors=True)
    merged = PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload()
    merged.save_pretrained(target, safe_serialization=True)
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(target)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for arg in ("base", "adapter", "target"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    merge(a.base, a.adapter, a.target)
