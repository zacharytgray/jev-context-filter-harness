"""a self-hosted vllm server: the sampling flags reach the request only when set, and the driver's local mode builds
its commands and refuses a launch when the server is down. nothing leaves the machine"""
import importlib.util
import json
import urllib.error
from pathlib import Path

import httpx
import pytest

from s1harness import cli
from s1harness import llm as L
from s1harness.jev import Trace

ROOT = Path(__file__).resolve().parent.parent
OK = {"id": "r1", "object": "chat.completion", "created": 0, "model": "m",
      "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi", "reasoning_content": "think"},
                   "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}
VLLM = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
        "chat_template_kwargs": {"enable_thinking": False}}


def sent(tmp_path, **kw) -> tuple[dict, L.LLM]:
    """the json body of one chat call"""
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=OK)

    llm = L.LLM("http://127.0.0.1:18000/v1", "local-model", Trace(tmp_path), **kw)
    llm.client = llm.client.copy(http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    llm.chat([{"role": "user", "content": "q"}])
    return seen[0], llm


def test_the_old_sampling_sends_the_same_body(tmp_path):
    body, llm = sent(tmp_path, provider="Novita", max_output=8192,
                     sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 64})
    assert set(body) == {"model", "messages", "max_tokens", "temperature", "top_p", "top_k", "provider"}
    assert llm.waits == L.HOSTED_WAITS
    body, llm = sent(tmp_path)
    assert set(body) == {"model", "messages"} and llm.waits == L.LOCAL_WAITS


def test_vllm_sampling_lands_in_the_body_only_when_set(tmp_path):
    body, llm = sent(tmp_path, max_output=8192, sampling=dict(VLLM))
    # min_p, top_k and chat_template_kwargs ride extra_body, which the sdk merges into the top level
    assert {k: body[k] for k in VLLM} == VLLM and "provider" not in body and body["max_tokens"] == 8192
    assert llm.gen == {"max_tokens": 8192, "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}
    assert llm.extra == {"extra_body": {"top_k": 20, "min_p": 0.0, "chat_template_kwargs": {"enable_thinking": False}}}
    # an extra body means hosted waits, as top_k alone did before
    assert llm.waits == L.HOSTED_WAITS
    # vllm's reasoning_content is logged
    [row] = [json.loads(x) for x in (tmp_path / "llm.jsonl").read_text().splitlines()]
    assert row["reasoning"] == "think"


def test_the_cli_passes_the_vllm_flags(tmp_path, monkeypatch):
    got = {}

    class Stop(Exception):
        pass

    def fake_llm(*args, **kw):
        got["args"] = args
        raise Stop

    monkeypatch.setattr(cli, "LLM", fake_llm)
    monkeypatch.setattr("sys.argv", ["s1h", "q", "--workdir", str(tmp_path), "--runs-dir", str(tmp_path / "runs"),
                                     "--base-url", "http://127.0.0.1:18000/v1", "--model", "local-model",
                                     "--temperature", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0",
                                     "--presence-penalty", "1.5",
                                     "--chat-template-kwargs", '{"enable_thinking": false}'])
    with pytest.raises(Stop):
        cli.main()
    assert got["args"][-1] == VLLM
    monkeypatch.setattr("sys.argv", ["s1h", "q", "--workdir", str(tmp_path), "--runs-dir", str(tmp_path / "runs"),
                                     "--base-url", "http://127.0.0.1:18000/v1", "--model", "local-model",
                                     "--temperature", "1.0"])
    with pytest.raises(Stop):
        cli.main()
    assert got["args"][-1] == {"temperature": 1.0}


# ---- the driver

def driver():
    spec = importlib.util.spec_from_file_location("driver", ROOT / "experiments" / "driver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CELL = {"arms": ["plain", "relevance"], "tasks": ["t1"], "start": 0, "repeats": 1, "workers": 1}


def local_manifest(**kw) -> dict:
    return {"block": "local", "tag": "local", "suite": "main", "model": "local-model", "note": "n",
            "manifest": "experiments/main/local_manifest.json", "max_output": 8192, "context_window_tokens": 65536,
            "endpoint": {"kind": "local", "base_url": "http://127.0.0.1:18000/v1", "api_key_env": None},
            "sampling": dict(VLLM), "stop_at": {"jev_input_tokens": 1000}, "cells": [CELL], **kw}


def test_the_driver_builds_a_local_command():
    D = driver()
    c = D.cmd(local_manifest(), CELL)
    assert c[0] == str(ROOT / ".venv" / "bin" / "python")
    flags = " ".join(c[1:])
    assert "--base-url http://127.0.0.1:18000/v1 --max-output 8192 --context-window 65536" in flags
    assert "--provider" not in c and "--api-key-env" not in c and "openrouter" not in flags
    assert ("--temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --presence-penalty 1.5 "
            '--chat-template-kwargs {"enable_thinking": false}') in flags
    m = local_manifest()
    m["endpoint"]["api_key_env"] = "VLLM_KEY"
    assert "--api-key-env VLLM_KEY" in " ".join(D.cmd(m, CELL))


def test_a_temperature_only_manifest_sends_no_other_sampling():
    c = driver().cmd(local_manifest(sampling={"temperature": 0.15}), CELL)
    assert "--temperature 0.15" in " ".join(c)
    assert not {"--top-p", "--top-k", "--min-p", "--presence-penalty", "--chat-template-kwargs"} & set(c)


def test_the_driver_keeps_the_openrouter_command():
    D = driver()
    mp = ROOT / "experiments" / "main" / "manifest_gemma4-31b-32k.json"
    if not mp.exists():
        pytest.skip("the manifests come with the data build")
    m = json.loads(mp.read_text())
    m["manifest"] = "experiments/main/manifest_gemma4-31b-32k.json"
    c = D.cmd(m, m["cells"][0])
    i = c.index("--base-url")
    assert c[i:i + 16] == ["--base-url", "https://openrouter.ai/api/v1", "--api-key-env", "OPENROUTER_API_KEY",
                           "--provider", "Novita", "--max-output", "8192", "--temperature", "1.0", "--top-p", "0.95",
                           "--top-k", "64", "--max-turns", "40"]
    assert not {"--context-window", "--min-p", "--presence-penalty", "--chat-template-kwargs"} & set(c)


def run_driver(D, tmp_path, monkeypatch, m: dict) -> list[dict]:
    monkeypatch.setattr(D, "WT", tmp_path)
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(m))
    log = tmp_path / "blocks.jsonl"
    monkeypatch.setattr("sys.argv", ["driver.py", str(mpath), str(log)])
    D.main()
    return [json.loads(x) for x in log.read_text().splitlines()]


def test_the_driver_refuses_when_the_endpoint_is_down(tmp_path, monkeypatch):
    D = driver()

    def down(*a, **kw):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(D.urllib.request, "urlopen", down)
    monkeypatch.setattr(D, "key", lambda: pytest.fail("a local block reads no openrouter key"))
    monkeypatch.setattr(D.subprocess, "Popen", lambda *a, **kw: pytest.fail("nothing launches"))
    with pytest.raises(SystemExit) as e:
        run_driver(D, tmp_path, monkeypatch, local_manifest())
    assert e.value.code == 1
    [row] = [json.loads(x) for x in (tmp_path / "blocks.jsonl").read_text().splitlines()]
    assert row["started"] is False and "did not answer" in row["why"]


def test_the_driver_refuses_another_model(tmp_path, monkeypatch):
    D = driver()
    monkeypatch.setattr(D, "models", lambda base_url, key_env: ["other-model"])
    monkeypatch.setattr(D.subprocess, "Popen", lambda *a, **kw: pytest.fail("nothing launches"))
    with pytest.raises(SystemExit):
        run_driver(D, tmp_path, monkeypatch, local_manifest())
    [row] = [json.loads(x) for x in (tmp_path / "blocks.jsonl").read_text().splitlines()]
    assert row["started"] is False and row["served"] == ["other-model"]


def test_a_local_block_launches_and_logs_the_same_rows(tmp_path, monkeypatch):
    D = driver()
    asked, launched = [], []

    class Done:
        def __init__(self, c, cwd):
            launched.append(c)

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(D, "models", lambda base_url, key_env: asked.append((base_url, key_env)) or ["local-model"])
    monkeypatch.setattr(D, "key", lambda: pytest.fail("a local block reads no openrouter key"))
    monkeypatch.setattr(D.subprocess, "Popen", Done)
    monkeypatch.setattr(D.time, "sleep", lambda s: pytest.fail("no sleep: no key to catch up"))
    rows = run_driver(D, tmp_path, monkeypatch, local_manifest())
    assert asked == [("http://127.0.0.1:18000/v1", None)] and len(launched) == 1
    assert rows[0]["started"] is True and rows[0]["served"] == ["local-model"]
    # the log ends with a done row: no openrouter spend for a local server, no jev tokens from the fake runs
    assert rows[-1]["done"] is True and rows[-1]["openrouter_usd"] == 0.0 and rows[-1]["jev_input_tokens"] == 0


@pytest.mark.parametrize("extra,want", [
    ([], {}),
    (["--min-p", "0", "--presence-penalty", "1.5", "--chat-template-kwargs", '{"enable_thinking": false}'],
     {"min_p": 0.0, "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": False}})])
def test_run_py_passes_and_records_the_vllm_flags(tmp_path, monkeypatch, extra, want):
    import sys
    from test_turns import argv, records, runner, suite
    exp = suite(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run.py", "fake", "--arms", "a", "--repeats", "1", "--tasks", "t1",
                                      "--no-task-env", "--note", "test", "--top-k", "20", *extra])
    runner.main()
    rec = records(exp)[0]
    got = argv(exp, rec)
    # a 0 is passed too, and the record's sampling says what was sent
    assert {f: got[got.index(f) + 1] for f in ("--min-p", "--presence-penalty", "--chat-template-kwargs")
            if f in got} == {"--" + k.replace("_", "-"): json.dumps(v) if isinstance(v, dict) else str(v)
                             for k, v in want.items()}
    assert rec["sampling"] == {"top_k": 20, **want}
