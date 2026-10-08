"""the agent llm's retries: the sdk retries nothing, and every retry is the harness's own, logged in
llm.jsonl. the provider is a fake on httpx's MockTransport, so nothing leaves the machine"""
import json

import httpx
import openai
import pytest

from s1harness import llm as L
from s1harness.jev import Trace

OK = {"id": "r1", "object": "chat.completion", "created": 0, "model": "m",
      "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}


def fake(tmp_path, monkeypatch, script, provider=None):
    """an llm whose provider answers from script, one step per request: a status code, or an exception to raise.
    returns the llm, the requests seen, and the waits slept"""
    seen, slept = [], []

    def handle(request):
        seen.append(request)
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        return httpx.Response(step, json=OK if step == 200 else {"error": {"message": f"status {step}"}})

    monkeypatch.setattr(L.time, "sleep", slept.append)
    llm = L.LLM("http://provider.test/v1", "m", Trace(tmp_path), provider=provider)
    # the real client with its own options, max_retries included, on a fake transport
    llm.client = llm.client.copy(http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    return llm, seen, slept


def rows(tmp_path):
    return [json.loads(x) for x in (tmp_path / "llm.jsonl").read_text().splitlines()]


def test_the_sdk_retries_nothing(tmp_path):
    for provider in (None, "novita"):
        assert L.LLM("http://provider.test/v1", "m", Trace(tmp_path), provider=provider).client.max_retries == 0
    assert L.HOSTED_WAITS == (15, 30, 60, 60, 120, 120, 240) and L.LOCAL_WAITS == (1, 2)


def test_a_429_and_a_5xx_are_retried_by_the_harness_and_logged(tmp_path, monkeypatch):
    llm, seen, slept = fake(tmp_path, monkeypatch, [429, 503, 200], provider="novita")
    msg, tokens = llm.chat([{"role": "user", "content": "q"}])
    assert msg == {"role": "assistant", "content": "hi"} and tokens == 5
    assert len(seen) == 3 and slept == [15, 30]  # each request once: no hidden sdk retry
    retry = [r for r in rows(tmp_path) if r["kind"] == "retry"]
    assert [(r["attempt"], r["status"], r["wait_s"], r["call_kind"]) for r in retry] == [
        (1, 429, 15, "turn"), (2, 503, 30, "turn")]
    assert "RateLimitError" in retry[0]["error"] and "InternalServerError" in retry[1]["error"]
    [turn] = [r for r in rows(tmp_path) if r["kind"] == "turn"]
    assert turn["retries"] == 2


def test_connection_errors_and_timeouts_are_retried(tmp_path, monkeypatch):
    llm, seen, slept = fake(tmp_path, monkeypatch, [httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), 200])
    llm.chat([{"role": "user", "content": "q"}])
    assert len(seen) == 3 and slept == [1, 2]
    assert [(r["kind"], r["status"], r["error"].split(":")[0]) for r in rows(tmp_path)[:2]] == [
        ("retry", None, "APIConnectionError"), ("retry", None, "APITimeoutError")]


def test_the_last_failure_is_logged_and_raised(tmp_path, monkeypatch):
    llm, seen, slept = fake(tmp_path, monkeypatch, [500, 502, 504])
    with pytest.raises(openai.InternalServerError):
        llm.chat([{"role": "user", "content": "q"}], kind="compaction")
    assert len(seen) == 3 and slept == [1, 2]
    assert [(r["kind"], r["attempt"], r["status"], r["wait_s"], r["call_kind"]) for r in rows(tmp_path)] == [
        ("retry", 1, 500, 1, "compaction"), ("retry", 2, 502, 2, "compaction"), ("error", 3, 504, None, "compaction")]


def test_other_errors_are_not_retried(tmp_path, monkeypatch):
    llm, seen, slept = fake(tmp_path, monkeypatch, [401], provider="novita")
    with pytest.raises(openai.AuthenticationError):
        llm.chat([{"role": "user", "content": "q"}])
    assert len(seen) == 1 and slept == [] and not (tmp_path / "llm.jsonl").read_text()
    def status(code):
        return openai.APIStatusError("x", response=httpx.Response(code, request=httpx.Request("POST", "http://x")),
                                     body=None)
    assert [L.retryable(status(c)) for c in (400, 404, 408, 409, 429, 500, 529)] == [
        False, False, True, True, True, True, True]
