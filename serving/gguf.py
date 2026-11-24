"""Compare two GGUF builds using one pinned llama-server binary and identical grammar."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import secrets
import socket
import subprocess
import time
from urllib.request import Request, urlopen
import numpy as np


@contextmanager
def server(binary, model):
    secret = secrets.token_hex(24)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            str(binary),
            "-m",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-t",
            "2",
            "-ngl",
            "0",
            "-c",
            "512",
            "--parallel",
            "1",
            "--api-key",
            secret,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(150):
            if process.poll() is not None:
                raise RuntimeError("llama-server stopped during startup")
            try:
                with urlopen(
                    Request(url + "/health", headers={"Authorization": "Bearer " + secret}), timeout=1
                ) as r:
                    if r.status == 200:
                        break
            except OSError:
                time.sleep(0.2)
        else:
            raise TimeoutError("Model startup timed out")
        yield url, secret
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def generate(url, secret, prompt):
    data = json.dumps(
        dict(
            prompt=prompt,
            grammar='root ::= " " [ABCD]',
            n_predict=4,
            temperature=0,
            cache_prompt=False,
            seed=23,
        )
    ).encode()
    request = Request(
        url + "/completion",
        data=data,
        headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=25) as response:
        return json.load(response)["content"].strip()


def benchmark(binary, model, rows):
    observations = []
    with server(binary, model) as (url, secret):
        generate(url, secret, rows[0]["prompt"])
        for row in rows:
            start = time.perf_counter()
            choice = generate(url, secret, row["prompt"])
            observations.append(
                dict(
                    id=row["id"], choice=choice, target=row["target"], ms=(time.perf_counter() - start) * 1000
                )
            )
    return dict(
        bytes=model.stat().st_size,
        sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        accuracy=sum(r["choice"] == r["target"] for r in observations) / len(rows),
        p95_ms=float(np.percentile([r["ms"] for r in observations], 95)),
        observations=observations,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("binary", "f16", "q4", "cases", "out"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    rows = json.loads(args.cases.read_text())
    results = {k: benchmark(args.binary, getattr(args, k), rows) for k in ("f16", "q4")}
    args.out.write_text(
        json.dumps(
            dict(
                engine="llama.cpp b6800",
                threads=2,
                clients=1,
                cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )
