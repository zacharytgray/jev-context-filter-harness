import json

import httpx
import pytest

from s1harness import messages as M
from s1harness import pipeline as P
from s1harness.jev import Jev, JevError, Trace, _well_formed
from s1harness.levers import Levers, load_levers
from s1harness.llm import ContextOverflow
from s1harness.tools import Tools, schemas

from fakes import CONFIGS, LEVERS_DIR, call, calls, events, go, reply, tool_messages, trace_rows

LONG_MD = "\n".join(["# Doc", ""] + [f"## Part {i}\n" + "\n".join(f"part {i} line {k}, some filler text to make it long"
                                                                    for k in range(20)) for i in range(12)])


# ---- levers


def test_exactly_four_lever_files_with_matching_switches():
    assert sorted(p.stem for p in LEVERS_DIR.iterdir()) == sorted(CONFIGS)
    assert CONFIGS == ("plain", "relevance", "injection", "relevance-injection")
    for name in CONFIGS:
        lv = load_levers(LEVERS_DIR / f"{name}.toml")
        assert lv.name == name
        on = {k for k in ("relevance", "injection") if getattr(lv, k)}
        assert on == (set() if name == "plain" else set(name.split("-"))), name
        assert (LEVERS_DIR / f"{name}.toml").read_text().startswith("# ")
    import tomllib
    arms = tomllib.loads((LEVERS_DIR.parent / "experiments" / "main" / "test.toml").read_text())["arms"]
    assert {k: v["levers"] for k, v in arms.items()} == {c: f"levers/{c}.toml" for c in CONFIGS}


def test_default_levers_are_the_plain_loop():
    assert Levers() == Levers(relevance=False, injection=False) and Levers().name == "plain"
    assert load_levers(None) == Levers()
    assert load_levers(LEVERS_DIR / "plain.toml") == Levers()


@pytest.mark.parametrize("text", ["[filters]\nrelevance = true\nquestion = true\n", "[tools]\nsearch = false\n",
                                  "[filters]\nrelevance = 1\n", "relevance = true\n", "[filters]\ncap = true\n"])
def test_unknown_keys_sections_and_non_bools_are_errors(tmp_path, text):
    f = tmp_path / "x.toml"
    f.write_text(text)
    with pytest.raises(ValueError):
        load_levers(f)


def test_overrides_use_the_same_rules():
    assert load_levers(LEVERS_DIR / "plain.toml", ["filters.relevance=true"]).relevance
    for bad in ("tools.search=true", "filters.cap=true", "filters.cap=false"):  # the cap is no switch
        with pytest.raises(ValueError):
            load_levers(None, [bad])


def test_the_configuration_name_follows_the_switches(tmp_path):
    assert load_levers(LEVERS_DIR / "plain.toml", ["filters.injection=true"]).name == "injection"
    assert load_levers(LEVERS_DIR / "relevance.toml", ["filters.injection=true"]).name == "relevance-injection"
    assert load_levers(LEVERS_DIR / "relevance-injection.toml", ["filters.relevance=false"]).name == "injection"
    f = tmp_path / "mine.toml"
    f.write_text("[filters]\nrelevance = true\n")
    assert load_levers(f).name == "relevance"
    assert Levers(relevance=True).to_dict() == {"name": "relevance", "relevance": True, "injection": False}
    res, _, _, run_dir = go(tmp_path, [reply("ok")],
                            config=load_levers(LEVERS_DIR / "plain.toml", ["filters.relevance=true"]))
    assert json.loads((run_dir / "levers.json").read_text())["name"] == "relevance"


# ---- the loop


def test_plain_loop_prompt_tools_and_records(tmp_path):
    res, llm, _, run_dir = go(tmp_path, [reply("all done")])
    assert (res.end, res.turns, res.final_message, res.rerun) == ("delivered", 1, "all done", False)
    assert llm.seen[0][0] == {"role": "system", "content": M.AGENT_PROMPT + "\n\n" + M.FINAL_PROMPT}
    names = [t["function"]["name"] for t in llm.tools]
    assert names == ["read", "list", "edit", "write", "bash"]
    bash = next(t["function"] for t in llm.tools if t["function"]["name"] == "bash")
    assert bash["description"] == "Run one shell command in the working directory. Use read and list to look at files."
    assert all("question" not in t["function"]["parameters"]["properties"] for t in llm.tools)
    result = json.loads((run_dir / "result.json").read_text())
    assert result == {"request": "do the thing", "final_message": "all done", "end": "delivered", "turns": 1,
                      "max_turns": 40}
    lv = json.loads((run_dir / "levers.json").read_text())
    assert {k: lv[k] for k in ("name", "relevance", "injection")} == Levers().to_dict() and "cap" not in lv
    assert lv["constants"]["cap_chars"] == 50000
    assert lv["constants"] == P.constants()
    assert lv["constants"]["relevance_threshold"] == 0.5 and lv["constants"]["injection_threshold"] == 0.5
    assert lv["constants"]["finder_wording"] == "inserted-framing" and lv["constants"]["injection_max_calls"] == 10 and lv["constants"]["jev_model"] == "jev-latest"
    assert json.loads((run_dir / "tools.json").read_text()) == llm.tools
    assert not (run_dir / "work_log.json").exists()


def test_relevance_schema_requires_a_question_on_read_and_bash():
    tools = {t["function"]["name"]: t["function"] for t in schemas(True)}
    assert tools["read"]["parameters"]["required"] == ["path", "question"]
    assert tools["bash"]["parameters"]["required"] == ["command", "question"]
    assert tools["read"]["description"] == P.READ_WITH_QUESTION
    assert tools["bash"]["description"].endswith(P.BASH_WITH_QUESTION)
    assert "question" not in tools["list"]["parameters"]["properties"]
    assert schemas(False) == schemas()  # schemas never share state
    assert "question" not in {t["function"]["name"]: t["function"] for t in schemas()}["read"]["parameters"]["properties"]


def test_max_turns_ends_the_run(tmp_path):
    res, _, _, run_dir = go(tmp_path, [calls(call("list", path="."))] * P.MAX_TURNS)
    assert (res.end, res.turns) == ("max turns", 40)
    # the default stays 40, and both records say so
    assert P.MAX_TURNS == 40 and P.constants()["max_turns"] == 40
    assert json.loads((run_dir / "levers.json").read_text())["constants"]["max_turns"] == 40
    assert json.loads((run_dir / "result.json").read_text())["max_turns"] == 40
    assert len(tool_messages(run_dir)) == 40


@pytest.mark.parametrize("limit", [3, 60])
def test_a_run_can_set_its_own_turn_limit(tmp_path, limit):
    res, llm, _, run_dir = go(tmp_path, [calls(call("list", path="."))] * limit, run_kw={"max_turns": limit})
    assert (res.end, res.turns, len(llm.seen)) == ("max turns", limit, limit)
    lv = json.loads((run_dir / "levers.json").read_text())
    assert lv["constants"] == {**P.constants(), "max_turns": limit}
    assert json.loads((run_dir / "result.json").read_text())["max_turns"] == limit
    assert P.MAX_TURNS == 40  # the module default does not move


def test_the_agent_never_sees_the_turn_limit(tmp_path):
    # the limit is the harness's, never part of what the agent sees
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    _, a, _, _ = go(tmp_path / "a", [calls(call("list", path="."))] * 2 + [reply("done")], run_kw={"max_turns": 40})
    _, b, _, _ = go(tmp_path / "b", [calls(call("list", path="."))] * 2 + [reply("done")], run_kw={"max_turns": 60})
    assert a.seen == b.seen and a.tools == b.tools
    assert "40" not in M.system_prompt() and "turn" not in M.system_prompt().lower()


def test_parse_errors_keep_their_wording(tmp_path):
    bad_json = {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{nope"}}
    not_obj = {"id": "c2", "type": "function", "function": {"name": "read", "arguments": "[1]"}}
    res, _, _, run_dir = go(tmp_path, [calls(bad_json, not_obj, call("search", "c3", pattern="x")), reply("ok")])
    assert tool_messages(run_dir) == ["Error: the arguments were not valid JSON.",
                                      "Error: the arguments must be a JSON object.",
                                      "Error: there is no tool named search."]
    assert [e["error"] for e in events(run_dir, "result")] == [True, True, True]


def test_relevance_needs_a_question_on_read_and_bash(tmp_path):
    res, _, jev, run_dir = go(tmp_path, [calls(call("read", "r", path="a.md"), call("bash", "b", command="ls"),
                                               call("read", "r2", path="a.md", question="  ")), reply("ok")],
                              config="relevance", files={"a.md": "hi\n"})
    assert tool_messages(run_dir) == ["Error: read needs a question.", "Error: bash needs a question.",
                                      "Error: read needs a question."]
    assert jev.bodies == []


def test_without_relevance_read_takes_no_question_and_bash_drops_it(tmp_path):
    res, _, _, run_dir = go(tmp_path, [calls(call("read", "r", path="a.md"), call("read", "r2", path="a.md", question="x"),
                                             call("bash", "b", command="echo hi", question="x")), reply("ok")],
                            files={"a.md": "hi\n"})
    got = tool_messages(run_dir)
    assert got[0] == "    1\thi"
    assert got[1].startswith("Error: bad arguments:")  # no question param without the relevance filter
    assert got[2] == "hi\n(exit code 0)"


def test_other_tools_are_not_filtered_and_log_a_result(tmp_path):
    big = "word " * 30000
    res, _, jev, run_dir = go(tmp_path, [calls(call("write", "w", path="big.txt", content=big),
                                               call("list", "l", path="."), call("ask_user", "u", question="?"),
                                               call("edit", "e", path="big.txt", old_text="x", new_text="y")),
                                         reply("ok")], config="relevance-injection")
    # ask_user is gone from every configuration, so a call to it is an unknown tool
    assert tool_messages(run_dir) == ["Wrote big.txt.", "big.txt", "Error: there is no tool named ask_user.",
                                      "Error: old_text matches 0 times in big.txt, it must match exactly once"]
    assert [(e["call_id"], e["tool"]) for e in events(run_dir, "result")] == [
        ("w", "write"), ("l", "list"), ("u", "ask_user"), ("e", "edit")]
    assert events(run_dir, "output") == [] and jev.bodies == []


def test_bash_exit_code_line_is_outside_the_pipeline(tmp_path):
    res, _, _, run_dir = go(tmp_path, [calls(call("bash", "b", command="printf 'a\\nb'; exit 3"),
                                             call("bash", "b2", command="true")), reply("ok")])
    assert tool_messages(run_dir) == ["a\nb\n(exit code 3)", "(exit code 0)"]
    out = events(run_dir, "output")
    assert [(o["raw_chars"], o["raw_lines"], o["entered_chars"]) for o in out] == [(3, 2, 17), (0, 0, 13)]
    assert out[0]["command"] == "printf 'a\\nb'; exit 3" and out[0]["shown"] == [[1, 2]]


def test_tool_errors_enter_as_they_are(tmp_path):
    res, _, jev, run_dir = go(tmp_path, [calls(call("read", "r", path="missing.md", question="q")), reply("ok")],
                              config="relevance-injection")
    assert tool_messages(run_dir) == ["Error: missing.md is not a file"]
    [o] = events(run_dir, "output")
    assert o["ok"] is False and o["entered_chars"] == len("Error: missing.md is not a file") and o["shown"] == []
    assert jev.bodies == []


def test_llm_crash_keeps_records_then_raises(tmp_path):
    with pytest.raises(RuntimeError):
        go(tmp_path, [RuntimeError("provider 400")])
    result = json.loads((tmp_path / "run" / "result.json").read_text())
    assert result["end"] == "error" and result["error"] == "RuntimeError: provider 400" and "rerun" not in result


def test_context_overflow_ends_as_overflow(tmp_path):
    with pytest.raises(ContextOverflow):
        go(tmp_path, [ContextOverflow("too big")])
    assert json.loads((tmp_path / "run" / "result.json").read_text())["end"] == "overflow"


# ---- a failed jev request is retried, then its filter falls back open


@pytest.mark.parametrize("tool", ["read", "bash"])
@pytest.mark.parametrize("config,fail,how,ref", [
    ("relevance", "relevance", "fail", "all kept"), ("relevance", "relevance", "malformed", "all kept"),
    ("injection", "injection", "fail", "plain"), ("injection", "injection", "malformed", "plain"),
    ("injection", "finder", "fail", "plain"), ("injection", "finder", "malformed", "plain"),
    ("relevance-injection", "relevance", "fail", "all kept, injection on"),
    ("relevance-injection", "injection", "fail", "relevance"),
    ("relevance-injection", "finder", "malformed", "relevance")])
def test_a_failed_request_falls_back_as_if_the_step_were_off(tmp_path, config, fail, how, ref, tool):
    q = {"question": "where is part 3?"} if "relevance" in config else {}
    doc = LONG_MD.replace("part 3 line 4", "NEEDLE PLANT part 3 line 4")
    c = call("read", "r1", path="a.md", **q) if tool == "read" else call("bash", "r1", command="cat a.md", **q)
    (tmp_path / "ref").mkdir()
    # the same call where the failing step is off: relevance keeps every chunk (a fake that says yes to all), or the
    # injection filter is off
    ref_config, ref_kw = {"all kept": ("relevance", {"word": ""}), "plain": ("plain", {}),
                          "all kept, injection on": ("relevance-injection", {"word": ""}),
                          "relevance": ("relevance", {})}[ref]
    _, _, _, ref_dir = go(tmp_path / "ref", [calls(c), reply("done")], config=ref_config, files={"a.md": doc},
                          **ref_kw)
    res, llm, jev, run_dir = go(tmp_path, [calls(c), reply("done")], config=config, files={"a.md": doc},
                                **{how: fail})
    assert (res.end, res.rerun, res.turns) == ("delivered", False, 2) and not llm.replies
    assert tool_messages(run_dir) == tool_messages(ref_dir)  # no new agent text
    assert "rerun" not in json.loads((run_dir / "result.json").read_text()) and not events(run_dir, "jev_error")
    bad = [r for r in trace_rows(run_dir) if r["kind"] == "request" and r.get("error")]
    falls = events(run_dir, "jev_fallback")
    assert bad and len(falls) == len(bad)
    assert all((r["call_id"], r["filter"], r["attempts"], len(r["errors"])) == ("r1", fail, 4, 4) for r in bad)
    # the first round asks 6 at a time, so the rows and events are matched by what they cover, not by order
    assert sorted((e["call_id"], e["filter"], e["chunks"], e["attempts"], e["error"]) for e in falls) == sorted(
        ("r1", fail, r["chunks"], 4, r["error"]) for r in bad)
    rows = [r for r in trace_rows(run_dir) if r["kind"] == "chunk" and r["filter"] == ("relevance" if fail ==
                                                                                        "relevance" else "injection")]
    fell = [r for r in rows if r.get("fallback")]
    assert fell and {x for e in falls for x in e["chunks"]} == {r["chunk"] for r in fell}
    if fail == "relevance":
        assert all(r["decision"] == "kept" and r["p"] is None for r in fell)
    elif fail == "injection":
        assert all(r["decision"] == "clear" and r["p"] is None for r in fell)
    else:
        [r] = fell
        assert (r["decision"], r["outcome"], r["hidden"], r["rounds"][0]["reason"]) == ("flagged", "fallback", [],
                                                                                        "fallback")


def test_failed_relevance_requests_fall_back_one_chunk_each(tmp_path):
    # one chunk per request, so each of the 12 fails on its own after 4 attempts
    res, _, jev, run_dir = go(tmp_path, [calls(call("read", "r1", path="a.md", question="q")), reply("x")],
                              config="relevance", files={"a.md": LONG_MD}, fail="relevance")
    reqs = [r for r in trace_rows(run_dir) if r["kind"] == "request"]
    assert len(reqs) == 12 and all(len(r["body"]["questions"]) == 1 and r["attempts"] == 4 for r in reqs)
    assert len(jev.bodies) == 48
    assert res.end == "delivered" and tool_messages(run_dir)[0].startswith("read: 12 of 12 chunks answer the question.")


def _jev(tmp_path, handler):
    trace = Trace(tmp_path)
    jev = Jev("jev-test", trace, api_key="k")
    jev.http = httpx.Client(transport=httpx.MockTransport(handler))
    jev.retry_waits = (0, 0, 0)
    return jev, trace


NOUL = {"q0": {"type": "noul", "instructions": "x", "criteria": {"true": "a", "false": "b"}}}


def test_jev_retries_any_failure_after_1_then_2_s(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr("s1harness.jev.time.sleep", waits.append)
    replies = [httpx.Response(429), httpx.Response(200, json={"answers": {"q0": {"noul": 2}}, "usage": {"input_tokens": 5}}),
               httpx.Response(200, json={"answers": {"q0": {"noul": 0.7}}, "usage": {"input_tokens": 7}})]
    jev, _ = _jev(tmp_path, lambda r: replies.pop(0))
    jev.retry_waits = Jev.retry_waits
    assert Jev.retry_waits == (1, 2, 4)
    assert jev.ask({}, NOUL, "relevance", "c1", [0]) == {"q0": {"noul": 0.7}}
    assert waits == [1, 2] and (jev.requests, jev.input_tokens) == (1, 12)
    [r] = [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert r["attempts"] == 3 and "error" not in r and "malformed" not in r and r["usage"] == {"input_tokens": 12}
    assert r["errors"][0].startswith("HTTPStatusError") and r["errors"][1] == "malformed answers: q0"
    assert r["answers"] == {"q0": {"noul": 0.7}}


def test_jev_gives_up_after_4_attempts(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr("s1harness.jev.time.sleep", waits.append)
    posts = []

    def handler(request):
        posts.append(request)
        raise httpx.ReadTimeout("slow", request=request)
    jev, _ = _jev(tmp_path, handler)
    jev.retry_waits = Jev.retry_waits
    with pytest.raises(JevError) as e:
        jev.ask({}, NOUL, "finder", "c1", [2])
    assert waits == [1, 2, 4] and len(posts) == 4
    assert (e.value.call_id, e.value.filter, e.value.attempts, e.value.chunks) == ("c1", "finder", 4, [2])
    [r] = [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert r["error"].startswith("ReadTimeout") and r["call_id"] == "c1" and r["filter"] == "finder"
    assert r["attempts"] == 4 and len(r["errors"]) == 4 and r["usage"] is None


@pytest.mark.parametrize("payload", [{"answers": {}}, {"answers": {"q0": {"noul": 1.5}}}, ["not", "an", "object"],
                                     {"answers": {"q0": {"noul": True}}}])
def test_jev_missing_or_malformed_answers_raise(tmp_path, payload):
    jev, _ = _jev(tmp_path, lambda r: httpx.Response(200, json=payload))
    with pytest.raises(JevError) as e:
        jev.ask({}, NOUL, "relevance", "c1")
    assert e.value.attempts == 4


def test_choice_answers_validate():
    q = {"type": "choice", "criteria": {"1": None, "2": None}}
    assert _well_formed({"probabilities": {"1": 0.4, "2": 0.6}}, q)
    assert not _well_formed({"probabilities": {"1": 1.0}}, q)
    assert not _well_formed({"noul": 0.5}, q)


# ---- the cli


def test_cli_defaults_to_the_plain_loop_and_exits_3_on_a_jev_error(tmp_path, monkeypatch, capsys):
    from s1harness import cli
    from s1harness.loop import RunResult
    seen = {}

    def fake_run(request, workdir, levers, llm, jev, trace, context_window, max_output, max_turns, compaction_budget):
        seen["levers"], seen["model"], seen["window"] = levers, jev.model, context_window
        seen["max_turns"], seen["budget"] = max_turns, compaction_budget
        return RunResult("", "jev error", 2, trace.run_dir, rerun=True)
    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["s1h", "hello", "--base-url", "https://openrouter.ai/api/v1",
                                     "--model", "test-model", "--workdir", str(tmp_path), "--runs-dir", str(tmp_path / "runs")])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 3
    assert seen["levers"] == Levers() and seen["model"] == "jev-latest" and seen["window"] == 262144
    assert seen["max_turns"] == 40 and seen["budget"] == 32768
    assert "end: jev error" in capsys.readouterr().out


def test_cli_takes_the_context_window_for_compaction(monkeypatch, capsys):
    from s1harness import cli
    monkeypatch.setattr("sys.argv", ["s1h", "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    assert "--context-window" in out and "--levers" in out


def test_cli_passes_max_turns_on(tmp_path, monkeypatch, capsys):
    from s1harness import cli
    from s1harness.loop import RunResult
    seen = {}

    def fake_run(*a, max_turns, **kw):
        seen["max_turns"] = max_turns
        return RunResult("ok", "delivered", 1, tmp_path)
    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["s1h", "hello", "--base-url", "https://openrouter.ai/api/v1",
                                     "--model", "test-model", "--workdir", str(tmp_path), "--runs-dir", str(tmp_path / "runs"),
                                     "--max-turns", "60"])
    cli.main()
    assert seen["max_turns"] == 60


def test_cli_passes_a_compaction_budget_on(tmp_path, monkeypatch, capsys):
    # a run's own budget, 16,384 for the 16K run set
    from s1harness import cli
    from s1harness.loop import RunResult
    seen = {}

    def fake_run(*a, compaction_budget, **kw):
        seen["budget"] = compaction_budget
        return RunResult("ok", "delivered", 1, tmp_path)
    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["s1h", "hello", "--base-url", "https://openrouter.ai/api/v1",
                                     "--model", "test-model", "--workdir", str(tmp_path), "--runs-dir", str(tmp_path / "runs"),
                                     "--compaction-budget", "16384"])
    cli.main()
    assert seen["budget"] == 16384


def test_tools_read_keeps_its_numbering(tmp_path):
    (tmp_path / "a.txt").write_text("x\ny\n")
    res = Tools(tmp_path).run("read", {"path": "a.txt", "start_line": 2})
    assert res.text == "    2\ty" and res.total == 2 and res.lines == [(2, "y")] and res.file_lines == ["x", "y"]
    assert Tools(tmp_path).run("read", {"path": "../x"}).text == "Error: ../x is outside the working directory"


# ---- relevance + injection with nothing flagged is relevance alone, to the byte


@pytest.mark.parametrize("word", ["NEEDLE", ""])  # "": relevance keeps every chunk, so the bash output passes the cap
def test_injection_that_flags_nothing_changes_nothing_the_agent_sees(tmp_path, word):
    head = "import os\nimport sys\n\nX = 1\n\n"
    py = head + "\n\n".join(f"def f{i}():\n" + "\n".join(f"    x{k} = {i} * {k}  # filler to make it long"
                                                       for k in range(30)) + ("\n    NEEDLE" if i == 7 else "")
                            for i in range(20)) + "\n"
    big = "\n".join(f"line {n} of a long log, NEEDLE at times, with filler text to pass the cap" for n in range(1500))
    files = {"m.py": py, "short.md": "# s\n\nNEEDLE here\n", "big.log": big, "a.md": LONG_MD}
    cs = [calls(call("read", "r1", path="m.py", question="where is NEEDLE?")),
          calls(call("read", "r2", path="short.md", question="q")),
          calls(call("read", "r3", path="a.md", start_line=5, end_line=200, question="where is part 3?")),
          calls(call("bash", "b1", command="cat big.log", question="NEEDLE?")),
          calls(call("bash", "b2", command="grep -n part *.md", question="part 3")),
          reply("done")]
    seen = {}
    for config in ("relevance", "relevance-injection"):
        (tmp_path / config).mkdir()
        res, llm, jev, run_dir = go(tmp_path / config, cs, config=config, files=files, word=word)
        assert res.end == "delivered"
        seen[config] = (llm.seen, llm.tools, tool_messages(run_dir))
    assert seen["relevance"] == seen["relevance-injection"]
    # the injection filter did run. with every chunk kept, it stopped past the cap and the cap cut the output
    assert any(r["kind"] == "request" and r["filter"] == "injection" for r in trace_rows(run_dir))
    if not word:
        assert events(run_dir, "output")[3]["skipped_past_cap"] > 0
        assert "[output cut at 50000 characters" in seen["relevance"][2][3]
