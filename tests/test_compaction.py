"""standard compaction with the fake llm. nothing calls a model"""
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import openai
import pytest

from fakes import FakeJev, FakeLLM, call, calls, events, go, levers, reply
from s1harness import compaction as C
from s1harness import messages as M
from s1harness.jev import Trace
from s1harness.llm import LLM, ContextOverflow, too_long
from s1harness.loop import run
from s1harness.view import load_run, render

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import run_card as RC  # noqa: E402
from metrics import run_metrics  # noqa: E402

BIG = "\n".join(f"row {i}: some tool output that fills the window up" for i in range(600))  # about 31k chars
SUMMARY = "the agent listed the files"


def roles_alternate(msgs: list[dict]) -> bool:
    """system, then one user message, then assistant and tool turns. only a call's tool results run together"""
    roles = [m["role"] for m in msgs]
    return roles[:2] == ["system", "user"] and roles.count("user") == 1 and \
        all(x != y or x == "tool" for x, y in zip(roles, roles[1:])) and \
        all(y != "tool" or x in ("assistant", "tool") for x, y in zip(roles, roles[1:]))


def script(*tail):
    """turn 1 lists, turn 2 reads big.txt and lists in one message, then tail"""
    return [calls(call("list", "c1", path=".")),
            calls(call("read", "c2", path="big.txt"), call("bash", "c3", command="echo hi")), *tail]


def counts(*turns, summary=500):
    """the fake provider's prompt_tokens: the next of turns for each agent call, summary for each compaction call"""
    left = list(turns)

    def f(messages):
        if messages[0]["content"] == C.SUMMARY_SYSTEM:
            return summary
        return left.pop(0)
    return f


def test_the_budget_is_32768_or_the_window_less_max_output_and_the_margin():
    # a fixed 32,768 on gemma 4 31b's window. a smaller window's room wins
    assert C.COMPACTION_BUDGET == 32768 and C.CONTEXT_WINDOW == 262144
    assert C.budget(262144, None) == 32768 and C.budget(100000, 8000) == 32768
    assert C.budget(40000, 8000) == 40000 - 8000 - 4096 and C.budget(30000, None) == 30000 - 4096
    # a run can set its own budget; the window's room still wins when smaller
    assert C.budget(262144, 8192, 16384) == 16384 and C.budget(20000, 8000, 16384) == 20000 - 8000 - 4096


@pytest.mark.parametrize("over", [0, 1])
def test_a_run_with_a_16384_budget_compacts_past_16384_and_records_it(tmp_path, over):
    # the 16K pass: turn 2's call reports 16,384 plus over. only past 16,384 does compaction run, before turn 3
    replies = script(reply(SUMMARY), reply("done")) if over else script(reply("done"))
    res, llm, _, run_dir = go(tmp_path, replies, files={"big.txt": BIG}, tokens=counts(1000, 16384 + over, 1000),
                              run_kw={"compaction_budget": 16384, "max_output": 8192})
    assert res.end == "delivered"
    assert llm.kinds == (["turn", "turn", "compaction", "turn"] if over else ["turn", "turn", "turn"])
    assert [(c["turn"], c["trigger_prompt_tokens"]) for c in events(run_dir, "compaction")] == \
        ([(3, 16385)] if over else [])
    lv = json.loads((run_dir / "levers.json").read_text())["compaction"]
    assert lv == {"context_window": 262144, "max_output": 8192, "margin": 4096, "compaction_budget": 16384,
                  "budget": 16384}


def test_no_character_estimate_is_left():
    # the trigger is the provider's exact prompt_tokens, never characters over a rate
    from s1harness import loop
    assert not hasattr(C, "CHARS_PER_TOKEN") and not hasattr(C, "tokens")
    src = inspect.getsource(loop)
    assert "C.tokens" not in src and "2.5" not in src and "def estimate" not in src


@pytest.mark.parametrize("over", [0, 1])
def test_compaction_fires_only_after_a_call_whose_prompt_passes_32768(tmp_path, over):
    # turn 2's call reports 32,768 plus over. only past 32,768 does compaction run, and only before turn 3
    replies = script(reply(SUMMARY), reply("done")) if over else script(reply("done"))
    res, llm, _, run_dir = go(tmp_path, replies, files={"big.txt": BIG}, tokens=counts(1000, 32768 + over, 1000))
    assert res.end == "delivered"
    assert llm.kinds == (["turn", "turn", "compaction", "turn"] if over else ["turn", "turn", "turn"])
    comps = events(run_dir, "compaction")
    assert [(c["turn"], c["trigger"], c["trigger_prompt_tokens"]) for c in comps] == \
        ([(3, "exact", 32769)] if over else [])
    assert all("estimate" not in c for c in comps)
    lv = json.loads((run_dir / "levers.json").read_text())["compaction"]
    assert (lv["context_window"], lv["compaction_budget"], lv["budget"]) == (262144, 32768, 32768)


def test_a_huge_new_output_does_not_compact_while_the_prompts_stay_at_the_budget(tmp_path):
    # about 45k characters enter on turn 2 and again on turn 3. at 2.5 characters a token that alone passed the
    # budget; the provider's counts are 32,768 at most, so every call goes ahead with no compaction
    huge = "\n".join(f"line {i}: " + "x" * 60 for i in range(600))
    replies = [calls(call("list", "c1", path=".")), calls(call("read", "c2", path="huge.txt")),
               calls(call("read", "c3", path="huge.txt")), reply("done")]
    res, llm, _, run_dir = go(tmp_path, replies, files={"huge.txt": huge},
                              tokens=counts(32768, 32768, 32768, 32768))
    assert 2 * len(huge) / 2.5 > 32768 and res.end == "delivered"
    assert llm.kinds == ["turn"] * 4 and not events(run_dir, "compaction")


def test_compaction_repeats_only_while_the_real_prompts_stay_over(tmp_path):
    # turn 2 and turn 3 report over the budget, turn 4 under it. the summary calls report far over and do not
    # count: only agent calls feed the trigger
    replies = [calls(call("list", "c1", path=".")), calls(call("list", "c2", path=".")), reply("S1"),
               calls(call("list", "c3", path=".")), reply("S2"), calls(call("list", "c4", path=".")),
               calls(call("list", "c5", path=".")), reply("done")]
    res, llm, _, run_dir = go(tmp_path, replies, tokens=counts(1000, 40000, 33000, 1000, 1000, 1000, summary=99999))
    assert res.end == "delivered"
    assert llm.kinds == ["turn", "turn", "compaction", "turn", "compaction", "turn", "turn", "turn"]
    comps = events(run_dir, "compaction")
    assert [(c["turn"], c["trigger"], c["trigger_prompt_tokens"], c["prompt_tokens"]) for c in comps] == \
        [(3, "exact", 40000, 99999), (4, "exact", 33000, 99999)]


def test_the_first_call_after_a_compaction_sets_the_next_trigger(tmp_path):
    # the prompt before the compaction was over; the next call's own count, under the budget, is what the trigger
    # reads after it
    replies = script(reply(SUMMARY), calls(call("list", "c4", path=".")), reply("done"))
    res, llm, _, run_dir = go(tmp_path, replies, files={"big.txt": BIG}, tokens=counts(1000, 50000, 2000, 2000))
    assert res.end == "delivered" and llm.kinds == ["turn", "turn", "compaction", "turn", "turn"]
    assert len(events(run_dir, "compaction")) == 1


@pytest.mark.parametrize("over", [0, 1])
def test_compaction_fires_only_past_the_threshold(tmp_path, over):
    # a smaller window's room is the budget: 10,000 here
    replies = script(reply(SUMMARY), reply("done")) if over else script(reply("done"))
    res, llm, _, run_dir = go(tmp_path, replies, files={"big.txt": BIG}, tokens=counts(1000, 10000 + over, 1000),
                              run_kw={"context_window": 10000 + C.MARGIN})
    assert res.end == "delivered"
    comps = events(run_dir, "compaction")
    assert llm.kinds == (["turn", "turn", "compaction", "turn"] if over else ["turn", "turn", "turn"])
    assert len(comps) == over
    if over:
        assert comps[0]["trigger_prompt_tokens"] == 10001 and comps[0]["turn"] == 3 and comps[0]["trigger"] == "exact"


def test_max_output_comes_off_the_window(tmp_path):
    _, llm, _, _ = go(tmp_path, script(reply(SUMMARY), reply("done")), files={"big.txt": BIG},
                      tokens=counts(1000, 10000, 1000),
                      run_kw={"context_window": 10000 + C.MARGIN + 100, "max_output": 101})
    assert llm.kinds.count("compaction") == 1


def test_a_call_with_no_usage_leaves_no_count(tmp_path):
    # llm.chat gives 0 when the provider sends no usage. that call feeds no trigger, so the next call goes
    # ahead however big the transcript; the one after an over count still compacts
    replies = [calls(call("list", "c1", path=".")), calls(call("read", "c2", path="big.txt")),
               calls(call("read", "c3", path="big.txt")), calls(call("list", "c4", path=".")), reply(SUMMARY),
               reply("done")]
    res, llm, _, run_dir = go(tmp_path, replies, files={"big.txt": BIG}, tokens=counts(40000, 1000, 0, 40000, 1000))
    assert res.end == "delivered"
    # turn 2: nothing to summarize yet. turn 3: 1,000 is under. turn 4: no count. turn 5: 40,000 is over
    assert llm.kinds == ["turn"] * 4 + ["compaction", "turn"]
    [e] = events(run_dir, "compaction")
    assert (e["turn"], e["trigger_prompt_tokens"]) == (5, 40000)


def test_the_real_client_gives_0_when_the_provider_sends_no_usage(tmp_path):
    trace = Trace(tmp_path)
    llm = LLM("http://localhost:1/v1", "m", trace)
    r = type("R", (), {"usage": None, "model": "m", "id": "x", "choices": [type("Ch", (), {
        "finish_reason": "stop", "message": type("M", (), {"content": "hi", "tool_calls": None})()})()]})()
    llm.client = type("X", (), {"chat": type("Y", (), {"completions": type("Z", (), {
        "create": lambda self, **_: r})()})()})()
    assert llm.chat([{"role": "user", "content": "q"}]) == ({"role": "assistant", "content": "hi"}, 0)


def test_the_newest_exchange_stays_verbatim_and_the_rest_becomes_one_summary(tmp_path):
    res, llm, _, run_dir = go(tmp_path, script(reply(SUMMARY), reply("done")), files={"big.txt": BIG},
                              request="count the rows", tokens=counts(1000, 40000, 1000, summary=1000))
    assert res.end == "delivered"
    sys_msg, req = llm.seen[0][0], llm.seen[0][1]
    history = json.loads((run_dir / "transcript.json").read_text())
    # the summary call: the fixed wording, the request, then only what lies between the request and the newest exchange
    summary_in = llm.seen[2]
    older = C.render(history[2:4])
    assert summary_in == [{"role": "system", "content": C.SUMMARY_SYSTEM},
                          {"role": "user", "content": C.SUMMARY_USER.format(request="count the rows", older=older)}]
    assert 'assistant calls list: {"path": "."}' in older and "row 599" not in summary_in[1]["content"]
    # after: the system prompt unchanged, the request and summary in one user message, then the newest exchange:
    # the assistant message with both its calls, and both results
    note = {"role": "user", "content": "count the rows\n\n" + C.SUMMARY_HEAD + SUMMARY}
    newest = history[5:8]
    assert [m["role"] for m in newest] == ["assistant", "tool", "tool"]
    assert [c["id"] for c in newest[0]["tool_calls"]] == ["c2", "c3"]
    assert [m["tool_call_id"] for m in newest[1:]] == ["c2", "c3"] and "row 599" in newest[1]["content"]
    assert llm.seen[3] == [sys_msg, note, *newest] and llm.seen[3][0] == sys_msg == llm.seen[0][0]
    assert req == {"role": "user", "content": "count the rows"} and roles_alternate(llm.seen[3])
    # transcript.json keeps everything, the note just before the exchange kept with it
    assert history[4] == note and history[:4] == llm.seen[1][:4]
    assert json.loads((run_dir / "context.json").read_text()) == [sys_msg, note, *newest, reply("done")]
    [e] = events(run_dir, "compaction")
    assert (e["at"], e["dropped"], e["kept"]) == (4, 0, 3) and "pieces" not in e
    assert (e["trigger"], e["trigger_prompt_tokens"]) == ("exact", 40000)
    assert e["transcript_chars_before"] == C.chars(history[:4] + newest)
    assert e["after"] == C.chars([sys_msg, note, *newest]) and e["summary_chars"] == len(SUMMARY)
    assert e["completion_tokens"] == len(json.dumps(reply(SUMMARY))) // 4 + 1 and e["prompt_tokens"] == 1000


def test_nothing_before_the_newest_exchange_means_no_compaction(tmp_path):
    # turn 1's prompt is over the budget, but before turn 2 the newest exchange is all there is: it stays and the
    # call goes ahead. before turn 3 turn 1's exchange is older, so it is summarized
    res, llm, _, run_dir = go(tmp_path, [calls(call("read", "c1", path="big.txt")), calls(call("list", "c2", path=".")),
                                         reply(SUMMARY), reply("done")],
                              files={"big.txt": BIG}, tokens=counts(40000, 40000, 1000))
    assert res.end == "delivered" and llm.kinds == ["turn", "turn", "compaction", "turn"]
    [e] = events(run_dir, "compaction")
    assert (e["turn"], e["trigger_prompt_tokens"]) == (3, 40000)


def test_one_summary_call_however_big_the_older_part_is(tmp_path):
    huge = "\n".join(f"line {i}: " + "x" * 60 for i in range(600))  # about 45k chars, one output under the cap
    replies = [calls(call("read", "c1", path="huge.txt")), calls(call("list", "c2", path=".")), reply(SUMMARY),
               reply("done")]
    res, llm, _, run_dir = go(tmp_path, replies, files={"huge.txt": huge}, tokens="chars",
                              run_kw={"context_window": 8000 + C.MARGIN})
    assert res.end == "delivered" and llm.kinds == ["turn", "turn", "compaction", "turn"]
    assert "line 0: " in llm.seen[2][1]["content"] and "line 599: " in llm.seen[2][1]["content"]
    assert not hasattr(C, "PIECES") and "piece" not in inspect.getsource(C).lower()


def test_a_provider_rejection_compacts_once_and_retries(tmp_path):
    res, llm, _, run_dir = go(tmp_path, [calls(call("list", "c1", path=".")), calls(call("list", "c2", path=".")),
                                         ContextOverflow("too long"), reply(SUMMARY), reply("done")])
    assert res.end == "delivered" and llm.kinds == ["turn", "turn", "turn", "compaction", "turn"]
    [e] = events(run_dir, "compaction")
    assert e["trigger"] == "rejected" and e["turn"] == 3
    assert llm.seen[4][1]["content"] == "do the thing\n\n" + C.SUMMARY_HEAD + SUMMARY
    assert [m["role"] for m in llm.seen[4]] == ["system", "user", "assistant", "tool"]
    assert llm.seen[4][2]["tool_calls"][0]["id"] == "c2"


def test_a_second_rejection_ends_the_run_as_overflow(tmp_path):
    with pytest.raises(ContextOverflow):
        go(tmp_path, [calls(call("list", "c1", path=".")), calls(call("list", "c2", path=".")),
                      ContextOverflow("too long"), reply(SUMMARY), ContextOverflow("still too long")])
    assert json.loads((tmp_path / "run" / "result.json").read_text())["end"] == "overflow"
    assert len(events(tmp_path / "run", "compaction")) == 1


def test_a_rejection_with_nothing_to_compact_ends_as_overflow(tmp_path):
    with pytest.raises(ContextOverflow):
        go(tmp_path, [calls(call("list", "c1", path=".")), ContextOverflow("too long")])
    assert json.loads((tmp_path / "run" / "result.json").read_text())["end"] == "overflow"
    assert not events(tmp_path / "run", "compaction")


def test_a_rejected_summary_call_drops_the_oldest_exchange_and_retries(tmp_path):
    replies = [calls(call("list", "c1", path=".")), calls(call("bash", "c2", command="echo two")),
               calls(call("bash", "c3", command="echo three")), calls(call("read", "c4", path="big.txt")),
               ContextOverflow("summary too long"), ContextOverflow("summary too long"), reply(SUMMARY), reply("done")]
    res, llm, _, run_dir = go(tmp_path, replies, files={"big.txt": BIG}, tokens=counts(1000, 1000, 1000, 6000, 1000),
                              run_kw={"context_window": 5000 + C.MARGIN})
    assert res.end == "delivered"
    assert llm.kinds == ["turn"] * 4 + ["compaction"] * 3 + ["turn"]
    ins = [x[1]["content"] for x in llm.seen[4:7]]
    assert all(x.startswith(C.SUMMARY_USER.split("{request}")[0] + "do the thing\n\n") for x in ins)
    assert ['"c1"' in x or 'list: {"path": "."}' in x for x in ins] == [True, False, False]
    assert ["echo two" in x for x in ins] == [True, True, False]
    assert ["echo three" in x for x in ins] == [True, True, True]
    assert not any("row 599" in x for x in ins)  # the newest exchange is never summarized
    [e] = events(run_dir, "compaction")
    assert e["dropped"] == 2 and e["kept"] == 2
    history = json.loads((run_dir / "transcript.json").read_text())
    assert llm.seen[-1][2:] == history[-3:-1] and llm.seen[-1][2]["tool_calls"][0]["id"] == "c4"
    assert roles_alternate(llm.seen[-1])


def test_the_fallback_drops_the_last_older_exchange_too(tmp_path):
    # no earlier summary: every older exchange is droppable, the last one included, before the run ends
    replies = [calls(call("list", "c1", path=".")), calls(call("bash", "c2", command="echo two")),
               calls(call("read", "c3", path="big.txt")),
               ContextOverflow("x"), ContextOverflow("x"), ContextOverflow("x")]
    work, run_dir = tmp_path / "work", tmp_path / "run"
    work.mkdir()
    run_dir.mkdir()
    (work / "big.txt").write_text(BIG)
    trace = Trace(run_dir)
    llm = FakeLLM(replies, tokens=counts(1000, 1000, 6000))
    with pytest.raises(ContextOverflow):
        run("do the thing", work, levers("plain"), llm, FakeJev(trace), trace, context_window=5000 + C.MARGIN)
    assert llm.kinds == ["turn"] * 3 + ["compaction"] * 3
    ins = [x[1]["content"] for x in llm.seen[3:6]]
    assert ["echo two" in x for x in ins] == [True, True, False]
    assert ['list: {"path": "."}' in x for x in ins] == [True, False, False]
    assert all("row 599" not in x and "do the thing" in x for x in ins)
    assert json.loads((run_dir / "result.json").read_text())["end"] == "overflow"


def test_the_fallback_keeps_an_earlier_summary_and_ends_as_overflow_when_only_it_is_left(tmp_path):
    replies = [calls(call("list", "c1", path=".")), calls(call("read", "c2", path="big.txt")), reply("S1"),
               calls(call("read", "c3", path="big.txt")),
               ContextOverflow("summary too long"), ContextOverflow("summary too long")]
    with pytest.raises(ContextOverflow):
        go(tmp_path, replies, files={"big.txt": BIG}, tokens=counts(1000, 6000, 6000),
           run_kw={"context_window": 5000 + C.MARGIN})
    run_dir = tmp_path / "run"
    assert json.loads((run_dir / "result.json").read_text())["end"] == "overflow"
    assert len(events(run_dir, "compaction")) == 1
    rows = [json.loads(x) for x in (run_dir / "trace.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "end" and rows[-1]["end"] == "overflow"


def test_the_fallback_summary_calls_carry_the_earlier_summary(tmp_path):
    replies = [calls(call("list", "c1", path=".")), calls(call("read", "c2", path="big.txt")), reply("S1"),
               calls(call("read", "c3", path="big.txt")),
               ContextOverflow("summary too long"), ContextOverflow("summary too long")]
    work = tmp_path / "work"
    work.mkdir()
    (work / "big.txt").write_text(BIG)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trace = Trace(run_dir)
    llm = FakeLLM(replies, tokens=counts(1000, 6000, 6000))
    with pytest.raises(ContextOverflow):
        run("do the thing", work, levers("plain"), llm, FakeJev(trace), trace, context_window=5000 + C.MARGIN)
    assert llm.kinds == ["turn", "turn", "compaction", "turn", "compaction", "compaction"]
    first, second = llm.seen[4][1]["content"], llm.seen[5][1]["content"]
    for x in (first, second):  # the request and the earlier summary are never dropped
        assert "do the thing\n\nuser:\n" + C.SUMMARY_HEAD + "S1" in x
    assert '"c2"' not in first and "big.txt" in first and "big.txt" not in second  # the exchange of c2, dropped
    assert "row 599" in first  # c2's output is older now; c3's is the newest exchange, never in the input
    assert first.count("row 599") == 1


def _err(status, text):
    r = httpx.Response(status, request=httpx.Request("POST", "http://x"))
    return (openai.BadRequestError if status == 400 else openai.APIStatusError)(text, response=r, body=None)


def test_a_context_length_error_from_the_provider_is_an_overflow(tmp_path):
    assert too_long(_err(400, "This model's maximum context length is 262144 tokens"))
    assert too_long(_err(413, "prompt is too long"))
    assert not too_long(_err(400, "invalid tool schema"))
    trace = Trace(tmp_path)
    llm = LLM("http://localhost:1/v1", "m", trace)

    class Stub:
        def __init__(self, e):
            self.e = e

        def create(self, **_):
            raise self.e
    llm.client = type("X", (), {"chat": type("Y", (), {"completions": Stub(_err(400, "context length exceeded"))})()})()
    with pytest.raises(ContextOverflow):
        llm.chat([{"role": "user", "content": "hi"}])
    llm.client.chat.completions = Stub(_err(400, "invalid tool schema"))
    with pytest.raises(openai.BadRequestError):
        llm.chat([{"role": "user", "content": "hi"}])


# ---- the records: metrics, the run card, the view


class RecordingLLM(FakeLLM):
    """the fake llm, writing llm.jsonl as the real one does. a compaction call gets "sum <k>", so the script holds
    only the turns"""

    def __init__(self, replies, trace):
        super().__init__(replies, tokens="chars")
        self.trace = trace
        self.sums = 0

    def chat(self, messages, tools=None, kind="turn"):
        if kind == "compaction":
            self.sums += 1
            self.replies.insert(0, reply(f"sum {self.sums}"))
        msg, p = super().chat(messages, tools, kind)
        self.trace.llm({"kind": kind, "latency_ms": 5, "n_messages": len(messages), "prompt_tokens": p,
                        "completion_tokens": self.completion_tokens, "message": msg})
        return msg, p


def recorded(tmp_path, replies, window, files):
    work, run_dir = tmp_path / "work", tmp_path / "run"
    work.mkdir()
    run_dir.mkdir()
    for k, v in files.items():
        (work / k).write_text(v)
    trace = Trace(run_dir)
    res = run("find the rows", work, levers("plain"), RecordingLLM(replies, trace), FakeJev(trace), trace,
              context_window=window)
    return res, run_dir


def two_compactions(tmp_path):
    a = "\n".join(f"alpha {i}: " + "a" * 50 for i in range(300))  # about 20k chars
    b = "\n".join(f"beta {i}: " + "b" * 50 for i in range(300))
    replies = [calls(call("list", "c0", path=".")), calls(call("read", "c1", path="a.txt")),
               calls(call("read", "c2", path="b.txt"), call("bash", "c3", command="echo hi")),
               calls(call("list", "c4", path=".")), reply("done")]
    return recorded(tmp_path, replies, 3000 + C.MARGIN, {"a.txt": a, "b.txt": b})


def test_roles_alternate_after_every_compaction(tmp_path):
    """chat templates like gemma's reject two user messages in a row. the tool replies to one call are the only run"""
    a = "\n".join(f"alpha {i}: " + "a" * 50 for i in range(300))
    b = "\n".join(f"beta {i}: " + "b" * 50 for i in range(300))
    work, run_dir = tmp_path / "work", tmp_path / "run"
    work.mkdir()
    run_dir.mkdir()
    (work / "a.txt").write_text(a)
    (work / "b.txt").write_text(b)
    trace = Trace(run_dir)
    llm = RecordingLLM([calls(call("list", "c0", path=".")), calls(call("read", "c1", path="a.txt")),
                        calls(call("read", "c2", path="b.txt"), call("bash", "c3", command="echo hi")),
                        calls(call("list", "c4", path=".")), reply("done")], trace)
    res = run("find the rows", work, levers("plain"), llm, FakeJev(trace), trace, context_window=3000 + C.MARGIN)
    assert res.end == "delivered" and len(events(run_dir, "compaction")) == 2
    turns = [m for m, k in zip(llm.seen, llm.kinds) if k == "turn"]
    after = [m for m in turns if any(C.SUMMARY_HEAD in (x.get("content") or "") for x in m)]
    assert len(after) >= 2
    for msgs in after:
        roles = [m["role"] for m in msgs]
        assert roles[:2] == ["system", "user"] and roles.count("user") == 1
        assert msgs[1]["content"].startswith("find the rows\n\n" + C.SUMMARY_HEAD)
        for x, y in zip(roles, roles[1:]):
            assert x != y or x == "tool", roles
    # the second summary call still sees the first summary
    starts = [i for i, k in enumerate(llm.kinds) if k == "compaction" and llm.kinds[i - 1] == "turn"]
    assert len(starts) == 2
    assert "user:\n" + C.SUMMARY_HEAD + "sum " in llm.seen[starts[1]][1]["content"]


def test_metrics_count_compaction_apart_from_the_agent(tmp_path):
    res, run_dir = two_compactions(tmp_path)
    assert res.end == "delivered"
    rows = [json.loads(x) for x in (run_dir / "llm.jsonl").read_text().splitlines()]
    turn = [r for r in rows if r["kind"] == "turn"]
    comp = [r for r in rows if r["kind"] == "compaction"]
    m = run_metrics(run_dir)
    assert m["compactions"] == 2 and len(comp) == 2 and m["llm_calls"] == len(turn) == 5
    assert m["llm_prompt_tokens"] == sum(r["prompt_tokens"] for r in turn)
    assert m["compaction_prompt_tokens"] == sum(r["prompt_tokens"] for r in comp) > 0
    assert m["compaction_completion_tokens"] == sum(r["completion_tokens"] for r in comp)
    assert m["total_prompt_tokens"] == m["llm_prompt_tokens"] + m["compaction_prompt_tokens"]
    assert m["total_completion_tokens"] == m["llm_completion_tokens"] + m["compaction_completion_tokens"]


def test_run_card_attribution_stays_exact_across_compactions(tmp_path):
    res, run_dir = two_compactions(tmp_path)
    r = RC.load(run_dir)
    assert r["warnings"] == [] and len(r["compactions"]) == 2
    total = run_metrics(run_dir)["llm_prompt_tokens"]
    assert r["total"] == total
    assert r["prefix_contrib"] + r["agent_contrib"] + r["tool_contrib"] + r["other_contrib"] + \
        r["summary_contrib"] == pytest.approx(total)
    assert r["summary_contrib"] > 0 and all(c["contrib"] >= 0 for c in r["calls"])
    cs = {c["id"]: c for c in r["calls"]}
    # an output kept in the newest exchange is re-sent by every call of the segment after it, until a summary
    # replaces it: c0 by turns 2 and 3, c1 by turn 3, c2 and c3 kept by the first compaction (turn 4), c4 by the
    # second (turn 5)
    assert [cs[k]["resent"] for k in ("c0", "c1", "c2", "c3", "c4")] == [2, 1, 1, 1, 1]
    assert cs["c1"]["contrib"] > 0 and cs["c0"]["contrib"] == pytest.approx(2 * cs["c0"]["tok"])
    comps = events(run_dir, "compaction")
    # turn 3's prompt holds a.txt and turn 4's b.txt, both past the 3,000 budget
    assert [(e["turn"], e["kept"], e["dropped"], e["trigger"]) for e in comps] == [(4, 3, 0, "exact"), (5, 2, 0, "exact")]
    text = RC.card(r)
    assert "4. compactions (2)" in text and "(e) compaction summaries" in text and f"sum: {total:,}." in text
    assert "in 2 compactions" in text
    assert "before turn 4 (exact): last prompt " in text and "estimate" not in text.split("4. compactions")[1]


def test_rollup_counts_compactions_and_costs_agent_plus_compaction(tmp_path):
    recs = [{"test": "t", "arm": "plain", "configuration": "plain", "task": "t1", "rep": k, "end": "delivered",
             "llm_prompt_tokens": 100, "llm_completion_tokens": 10, "total_prompt_tokens": 100 + 50 * k,
             "total_completion_tokens": 10 + 5 * k, "compactions": k} for k in range(3)]
    (tmp_path / "t" / "plain").mkdir(parents=True)
    (tmp_path / "t" / "plain" / "records.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))
    R = RC.rollup(tmp_path)
    [row] = R["rows"]
    assert row["compactions"] == 3 and row["prompt_tokens"] == 100
    assert row["cost_usd"] == pytest.approx((150 * 0.14 + 15 * 0.40) / 1e6)
    assert R["total_cost_usd"] == pytest.approx((450 * 0.14 + 45 * 0.40) / 1e6)
    assert "| compactions |" in RC.rollup_text(R)


def test_view_embeds_the_compaction_event(tmp_path):
    res, run_dir = two_compactions(tmp_path)
    page = render(load_run(run_dir))
    assert '"event":"compaction"' in page
    if shutil.which("node") is None:
        return
    from test_view import DOM
    (tmp_path / "view.html").write_text(page)
    (tmp_path / "dom.js").write_text(DOM)
    out = subprocess.run(["node", str(tmp_path / "dom.js"), str(tmp_path / "view.html")], capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.count("Compaction before turn") == 2 and "[summary of earlier work]\nsum " in out.stdout
    assert "Turn 5" in out.stdout
    assert out.stdout.count("the last prompt passed the budget") == 2 and "estimate" not in out.stdout


def test_levers_json_records_the_window(tmp_path):
    _, _, _, run_dir = go(tmp_path, [reply("ok")], run_kw={"context_window": 100000, "max_output": 8000})
    lv = json.loads((run_dir / "levers.json").read_text())
    assert lv["compaction"] == {"context_window": 100000, "max_output": 8000, "margin": 4096,
                                "compaction_budget": 32768, "budget": 32768}
    assert not (run_dir / "context.json").exists() and M.system_prompt()
