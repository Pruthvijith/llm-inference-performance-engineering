import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .runtime import Worker


def summarise(measures, elapsed, hourly_cost=None):
    if not measures or elapsed <= 0:
        raise ValueError("Positive duration and observations required")
    report = {
        "n": len(measures),
        "elapsed_s": elapsed,
        "requests_per_s": len(measures) / elapsed,
        "tokens_per_s": sum(m["tokens"] for m in measures) / elapsed,
        "accuracy": sum(m["target"] == m["choice"] for m in measures) / len(measures),
    }
    for metric in ("ttft_ms", "latency_ms", "decode_gaps_ms"):
        vals = np.asarray([m[metric] for m in measures]).flatten()
        report[metric] = {f"p{q}": float(np.percentile(vals, q)) for q in (50, 95, 99)} if len(vals) else {}
    if hourly_cost is not None:
        if hourly_cost < 0:
            raise ValueError("Cost must be positive or zero")
        report["estimated_cost_per_1000"] = hourly_cost * elapsed / 3600 * 1000 / len(measures)
    return report


def run(worker, rows, concurrency, tokens):
    if not 1 <= concurrency <= 4:
        raise ValueError("Expected 1 to 4 clients")
    worker.infer(rows[0]["prompt"], tokens)

    def request(row):
        return {"id": row["id"], "target": row["target"], **worker.infer(row["prompt"], tokens)}

    with ThreadPoolExecutor(concurrency) as pool:
        start = time.perf_counter()
        measures = list(pool.map(request, rows))
        elapsed = time.perf_counter() - start
    buffer = io.BytesIO()
    torch.save(worker.model.state_dict(), buffer)
    return {**summarise(measures, elapsed), "state_dict_bytes": buffer.tell(), "requests": measures}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    rows = json.loads(args.cases.read_text())
    reports = {}
    for precision in ("fp32", "int8"):
        worker = Worker(args.model, args.revision, args.adapter, precision)
        reports[precision] = run(worker, rows, 2, 3)
        del worker
    args.out.write_text(
        json.dumps(
            dict(
                torch=torch.__version__,
                machine=platform.machine(),
                threads=args.threads,
                revision=args.revision,
                adapter_sha256=hashlib.sha256(
                    (args.adapter / "adapter_model.safetensors").read_bytes()
                ).hexdigest()
                if args.adapter
                else None,
                cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                clients=2,
                fixed_decode_tokens=3,
                reports=reports,
            ),
            indent=2,
        )
        + "\n"
    )
