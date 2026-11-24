# Private inference performance

Measure how compression changes decision quality and serving latency on the same trained adapter. Two paired CPU studies cover PyTorch dynamic INT8 and llama.cpp GGUF quantisation.

Independent, AI-assisted portfolio research milestone · v0.1.0 · Python 3.12 · CPU reference environment.

## Measured outcome

PyTorch FP32 and INT8 reached 14/48 and 10/48 correct. P95 worker latency was 657.2 and 207.3 ms. Serialised state sizes were 538,167,007 and 248,137,617 bytes. GGUF F16 and Q4_K_M reached 14/48 and 13/48 with file sizes of 270,885,312 and 105,453,504 bytes. llama.cpp used fallback quantisation on 180 of 211 quantised tensors due to matrix shape. GGUF Q4_K_M HTTP P95 was 190.9 ms versus 243.0 ms for F16. Results apply to the recorded hardware, model and load. See all raw observations under reports/.

## Architecture and data flow

The serving worker loads a locally staged base and optional PEFT adapter. It merges adapters, optionally quantises linear layers, serialises CPU execution and bounds the number of admitted requests. Clients measure time to first token, inter-token gaps and total latency. A separate harness starts a loopback llama-server with an ephemeral bearer token and compares two GGUF files.

## Dataset and model usage contract

48 held-out prompts from the same DPO lab experiment. The PyTorch test uses two concurrent clients and exactly three decoded tokens per request, plus a single warm-up. Quality uses the restricted A–D next-token choice from those same logits. Timings include the worker queue and exclude HTTP for this test. llama.cpp quality uses one HTTP client and a grammar for the four choices.

The supported purpose is offline research and human-reviewed engineering decision support. Aviation design approval, operational aircraft decisions and regulated certification are outside the validated scope.

## Reproduce

Run from this checkout on Linux x86-64. The reference CPU was an AMD EPYC 9V74. The portfolio index includes a downloader for the pinned models and UCI input.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install -c constraints-cpu.txt -r requirements-dev.txt
python -m serving.benchmark --model "$MODEL_DIRECTORY" --revision "$MODEL_REVISION" --adapter "$ADAPTER_PATH" --cases ../engineering-llm-evaluation-framework/examples/base.json --out work-cpu.json
```

## Testing and deployment

```bash
python -m pytest --cov=serving --cov-report=term-missing
python -m ruff check .
python -m mypy serving
```

Set MODEL_DIRECTORY to the staged SmolLM2 snapshot, MODEL_REVISION to its commit, and ADAPTER_PATH to the DPO adapter. Run python -m serving.export "$MODEL_DIRECTORY" "$ADAPTER_PATH" work/merged. With llama.cpp b6800 and its conversion dependencies, run convert_hf_to_gguf.py work/merged --outfile work/f16.gguf --outtype f16, followed by llama-quantize work/f16.gguf work/q4.gguf Q4_K_M 2. Run python -m serving.gguf /path/to/llama-server work/f16.gguf work/q4.gguf ../engineering-llm-evaluation-framework/examples/base.json work-gguf.json. For CPU serving, set MODEL_PATH, MODEL_REVISION, SERVING_SECRET and optional ADAPTER_PATH, then run python -m uvicorn serving.api:create_app --factory --host 127.0.0.1. POST JSON prompt and tokens to /v1/complete with a bearer token. Docker Compose examples live in deployments/. The AWQ GPU profile requires an operator-staged model and image digest. GPU execution remains unverified.

## Operations and next validation

HTTP 429 means admission capacity is exhausted. Back off with jitter. Monitor queue-inclusive P95, quality and input length together. For GGUF failures inspect engine build, tensor shape and prompt grammar. Revert to the previous file digest when a smaller model misses the quality floor.

See [decisions and interview guide](docs/review.md) and the measured files in [reports](reports/). Hosted CI and production qualification require their own evidence. For dependencies and known advisories see the portfolio index security report.

## Provenance and licensing

Original code and original fixtures use Apache-2.0. Upstream models, data and third-party executables retain their own licences. External weights and source archives are fetched separately. References follow.

- https://github.com/ggml-org/llama.cpp/releases/tag/b6800
- https://docs.vllm.ai/en/latest/features/quantization/

## Local verification

5 tests passed. Statement coverage of the serving package was 90.91%. Ruff and mypy passed. See [verification record](reports/verification.json).
