"""Bounded, serial CPU worker with an optional dynamic INT8 model."""

import hmac
import os
import threading
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class Busy(Exception):
    """All admission slots are occupied."""


def authorised(candidate: str, secret: str) -> bool:
    return len(secret) >= 24 and hmac.compare_digest(candidate.encode(), ("Bearer " + secret).encode())


class Worker:
    def __init__(self, model_id: str, revision: str, adapter: Path | None, precision: str, slots: int = 4):
        if precision not in ("fp32", "int8") or slots < 1:
            raise ValueError("Invalid worker settings")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float32,
        ).eval()
        if adapter:
            model = PeftModel.from_pretrained(model, adapter, local_files_only=True).merge_and_unload().eval()
        if precision == "int8":
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
            )
        self.model = model
        self.ids = [self.tokenizer.encode(" " + c, add_special_tokens=False) for c in "ABCD"]
        if any(len(ids) != 1 for ids in self.ids):
            raise ValueError("Invalid choice tokenisation")
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(slots)

    @torch.inference_mode()
    def infer(self, prompt: str, tokens: int):
        if not prompt.strip() or len(prompt) > 1500 or not 1 <= tokens <= 12:
            raise ValueError("Invalid request bounds")
        arrival = time.perf_counter()
        if not self.slots.acquire(blocking=False):
            raise Busy()
        try:
            with self.lock:
                encoded = self.tokenizer(prompt, return_tensors="pt").input_ids
                if encoded.shape[1] + tokens > self.model.config.max_position_embeddings:
                    raise ValueError("Context length exceeded")
                stream = []
                cache = None
                response = []
                choice = ""
                for step in range(tokens):
                    out = self.model(input_ids=encoded, past_key_values=cache, use_cache=True)
                    logits = out.logits[0, -1]
                    if step == 0:
                        choice = "ABCD"[int(logits[[t[0] for t in self.ids]].argmax())]
                    encoded = logits.argmax().reshape(1, 1)
                    response.append(int(encoded.item()))
                    cache = out.past_key_values
                    stream.append(time.perf_counter())
                return dict(
                    text=self.tokenizer.decode(response),
                    choice=choice,
                    tokens=len(response),
                    ttft_ms=1000 * (stream[0] - arrival),
                    decode_gaps_ms=[1000 * (b - a) for a, b in zip(stream, stream[1:])],
                    latency_ms=1000 * (stream[-1] - arrival),
                )
        finally:
            self.slots.release()


def from_environment():
    torch.set_num_threads(int(os.getenv("CPU_THREADS", "2")))
    return Worker(
        os.environ["MODEL_PATH"],
        os.environ["MODEL_REVISION"],
        Path(os.environ["ADAPTER_PATH"]) if os.getenv("ADAPTER_PATH") else None,
        os.getenv("PRECISION", "fp32"),
    )
