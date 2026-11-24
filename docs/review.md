# Design decision and review guide

## Decision record 001

Keep each engine comparison paired. The PyTorch worker serialises execution and measures queueing under contention. The llama.cpp study uses one client and constrained generation. Comparing cross-engine latencies would conflate workload and backend. Report state-file bytes separately from process RSS and GPU VRAM. The pricing helper takes an explicit hourly rate, and no paid rate was assumed.

## Operational response

HTTP 429 means admission capacity is exhausted. Back off with jitter. Monitor queue-inclusive P95, quality and input length together. For GGUF failures inspect engine build, tensor shape and prompt grammar. Revert to the previous file digest when a smaller model misses the quality floor.

## Interview practice

Use these answers after reproducing the experiment. Describe the work as an AI-assisted independent portfolio project.

### Would you deploy INT8?

The latency improvement cost four correct answers on 48 cases. The starting model already failed qualification. I would retain the benchmark and withhold both builds.

### Why can a smaller file still require substantial RAM?

The runtime also holds unquantised layers, activations and the KV cache. Concurrent sequences and context length change the memory profile. A disk-size measurement cannot establish peak RAM.

### How would you size an on-prem GPU?

Measure prompt and decode separately at the real concurrency and context-length distribution. Add KV-cache memory, transient activations and operational reserve before selecting tensor parallelism.

