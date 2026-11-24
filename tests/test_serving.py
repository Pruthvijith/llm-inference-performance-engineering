import pytest
import torch
from fastapi.testclient import TestClient
from serving.api import create_app
from serving.runtime import Worker, Busy, authorised
from serving.benchmark import summarise, run
from serving.export import merge
from peft import LoraConfig, get_peft_model
from transformers import LlamaForCausalLM, LlamaConfig, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers


def test_cpu_modes_and_export(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    vocab = {s: i for i, s in enumerate(["<unk>", "<eos>", "A", "B", "C", "D"])}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, eos_token="<eos>", unk_token="<unk>")
    base = tmp_path / "base"
    tok.save_pretrained(base)
    m = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=6,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
        )
    )
    m.save_pretrained(base)
    ad = tmp_path / "adapter"
    get_peft_model(m, LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM")).save_pretrained(ad)
    merge(base, ad, tmp_path / "merged")
    for precision in ("fp32", "int8"):
        w = Worker(str(base), "main", ad, precision, 1)
        m = w.infer("A B", 2)
        assert m["tokens"] == 2 and len(m["decode_gaps_ms"]) == 1
        out = run(w, [dict(id="1", prompt="A", target="A")], 1, 2)
        assert out["requests_per_s"] > 0 and out["state_dict_bytes"] > 0
        for prompt, tokens in [("", 2), ("A " * 35, 1), ("A", 13)]:
            with pytest.raises(ValueError):
                w.infer(prompt, tokens)
        w.slots.acquire()
        with pytest.raises(Busy):
            w.infer("A", 1)
        w.slots.release()

    import json
    import runpy
    import sys

    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [{"id": "one", "target": "A", "prompt": "A B"}, {"id": "two", "target": "B", "prompt": "B A"}]
        )
    )
    output = tmp_path / "paired.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            str(base),
            "--revision",
            "main",
            "--adapter",
            str(ad),
            "--cases",
            str(cases_path),
            "--out",
            str(output),
            "--threads",
            "1",
        ],
    )
    runpy.run_module("serving.benchmark", run_name="__main__")
    comparison = json.loads(output.read_text())
    assert comparison["reports"]["int8"]["n"] == comparison["reports"]["fp32"]["n"] == 2
    assert comparison["clients"] == 2 and len(comparison["adapter_sha256"]) == 64


def test_api():
    secret = "x" * 32

    class Stub:
        def infer(self, prompt, tokens):
            if prompt == "busy":
                raise Busy()
            if prompt == "invalid":
                raise ValueError()
            return {"tokens": tokens}

    client = TestClient(create_app(Stub(), secret))
    auth = {"Authorization": "Bearer " + secret}
    assert client.post("/v1/complete", json={"prompt": "ok"}).status_code == 401
    for prompt, status in [("ok", 200), ("busy", 429), ("invalid", 422), ("", 422)]:
        assert client.post("/v1/complete", headers=auth, json={"prompt": prompt}).status_code == status
    with pytest.raises(ValueError):
        create_app(Stub(), "short")
    assert not authorised("Bearer test", "test")


def test_metrics_and_bounds():
    sample = {"tokens": 2, "target": "A", "choice": "B", "ttft_ms": 2, "latency_ms": 4, "decode_gaps_ms": [2]}
    result = summarise([sample], 1, 3.6)
    assert result["estimated_cost_per_1000"] == 1 and result["accuracy"] == 0
    with pytest.raises(ValueError):
        summarise([], 1)
    with pytest.raises(ValueError):
        summarise([sample], 1, -1)
    with pytest.raises(ValueError):
        run(None, [], 5, 1)
    with pytest.raises(ValueError):
        Worker("unused", "main", None, "bogus")


def test_service_lifecycle_and_gguf_measurement(tmp_path, monkeypatch):
    import json
    from io import BytesIO
    from serving import gguf

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.stopped = False

        def poll(self):
            return None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout):
            assert self.stopped

    class Reply(BytesIO):
        status = 200

    def handle(request, timeout):
        assert request.full_url.startswith("http://127.0.0.1:")
        assert request.get_header("Authorization").startswith("Bearer ")
        if request.data:
            assert json.loads(request.data)["cache_prompt"] is False
        return Reply(b'{"content":" A"}')

    monkeypatch.setattr(gguf.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(gguf, "urlopen", handle)
    model = tmp_path / "f.gguf"
    model.write_bytes(b"test-artifact")
    report = gguf.benchmark(tmp_path / "local-engine", model, [dict(id="a", target="A", prompt="record")])
    assert report["accuracy"] == 1 and report["bytes"] == 13


def test_invalid_launch_and_configuration(monkeypatch):
    from serving import gguf, runtime

    class Failed:
        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return 1

        def terminate(self):
            pass

        def wait(self, timeout):
            pass

    monkeypatch.setattr(gguf.subprocess, "Popen", Failed)
    with pytest.raises(RuntimeError, match="startup"):
        with gguf.server("binary", "fixture"):
            pass
    monkeypatch.setenv("MODEL_PATH", "fixture")
    monkeypatch.setenv("MODEL_REVISION", "rev")
    monkeypatch.setenv("ADAPTER_PATH", "adapter")
    monkeypatch.setattr(runtime, "Worker", lambda *args: args)
    base, revision, adapter, precision = runtime.from_environment()
    assert base == "fixture" and revision == "rev" and adapter.name == "adapter" and precision == "fp32"
