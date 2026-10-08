"""the per-run turn limit in the runner and the driver, and their dry runs. nothing paid"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import run as runner  # noqa: E402

# ---- the runner

FAKE_CLI = """import json, sys
from pathlib import Path
runs = Path(sys.argv[sys.argv.index("--runs-dir") + 1]) / "20260930-000000-000000"
runs.mkdir(parents=True)
(runs / "argv.json").write_text(json.dumps(sys.argv[1:]))
(runs / "result.json").write_text(json.dumps({"request": "x", "final_message": "", "end": "delivered", "turns": 44}))
"""


def suite(tmp_path, monkeypatch, toml_max=None):
    exp = tmp_path / "exp"
    (exp / "fake").mkdir(parents=True)
    (exp / "fake" / "test.toml").write_text((f"max_turns = {toml_max}\n" if toml_max else "") +
                                            '[arms.a]\nlevers = "levers/plain.toml"\n')
    for t in ("t1", "t2"):
        (exp / "fake" / "tasks" / t).mkdir(parents=True)
        (exp / "fake" / "tasks" / t / "task.toml").write_text('request = "x"\n')
    (tmp_path / "fake_cli.py").write_text(FAKE_CLI)
    monkeypatch.setattr(runner, "EXP", exp)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(runner, "CLI", [sys.executable, str(tmp_path / "fake_cli.py")])
    monkeypatch.setattr(runner, "code_version", lambda test: {"commit": "x", "dirty": False})
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: type("R", (), {"stdout": ""})())
    monkeypatch.setattr(runner, "prepare", lambda task, work, setup=True: work.mkdir(parents=True))
    monkeypatch.setattr(runner, "check", lambda task, work, run_dir=None: {"score": 0.5})
    return exp


def records(exp: Path, tag: str | None = None) -> list[dict]:
    f = exp / (f"runs-{tag}" if tag else "runs") / "fake" / "a" / "records.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines()]


def argv(exp: Path, rec: dict) -> list[str]:
    return json.loads((exp.parent / rec["run_dir"] / "argv.json").read_text())


@pytest.mark.parametrize("toml_max,flag,want", [(None, None, None), (60, None, "60"), (60, "45", "45"),
                                                (None, "45", "45")])
def test_run_py_takes_max_turns_from_the_command_line_then_the_test_toml(tmp_path, monkeypatch, toml_max, flag, want):
    exp = suite(tmp_path, monkeypatch, toml_max)
    monkeypatch.setattr(sys, "argv", ["run.py", "fake", "--arms", "a", "--repeats", "1", "--tasks", "t1",
                                      "--no-task-env", "--note", "test", *(["--max-turns", flag] if flag else [])])
    runner.main()
    rec = records(exp)[0]
    got = argv(exp, rec)
    assert (got[got.index("--max-turns") + 1] if "--max-turns" in got else None) == want
    assert rec.get("max_turns") == (int(want) if want else None)


@pytest.mark.parametrize("flag", [None, "16384"])
def test_run_py_passes_a_compaction_budget_and_records_it(tmp_path, monkeypatch, flag):
    # unset leaves the cli's 32,768 and puts nothing in the record
    exp = suite(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["run.py", "fake", "--arms", "a", "--repeats", "1", "--tasks", "t1",
                                      "--no-task-env", "--note", "test",
                                      *(["--compaction-budget", flag] if flag else [])])
    runner.main()
    rec = records(exp)[0]
    got = argv(exp, rec)
    assert (got[got.index("--compaction-budget") + 1] if "--compaction-budget" in got else None) == flag
    assert rec.get("compaction_budget") == (int(flag) if flag else None)


# ---- the driver and the manifest

def driver():
    spec = importlib.util.spec_from_file_location("driver", ROOT / "experiments" / "driver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# a manifest with the keys the driver reads
MANIFEST = {"block": "b", "suite": "main", "tag": "x", "model": "m", "provider": "p", "max_output": 100, "note": "n",
            "manifest": "experiments/x_manifest.json", "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
            "cells": [{"arms": ["plain"], "tasks": ["t1"], "start": 0, "repeats": 1, "workers": 1}]}


def test_the_driver_passes_max_turns_from_the_manifest():
    d = driver()
    for extra, want in (({"max_turns": 60}, "60"), ({"max_turns": 40}, "40"), ({}, None)):
        m = {**MANIFEST, **extra}
        cmd = d.cmd(m, m["cells"][0])
        assert (cmd[cmd.index("--max-turns") + 1] if "--max-turns" in cmd else None) == want, extra
        assert cmd[-1].endswith("(experiments/x_manifest.json)")


def test_the_driver_passes_a_compaction_budget_only_when_the_manifest_sets_one():
    d = driver()
    m = MANIFEST
    assert "--compaction-budget" not in d.cmd(m, m["cells"][0])  # no key: the harness's 32,768
    cmd = d.cmd({**m, "compaction_budget": 16384}, m["cells"][0])
    assert cmd[cmd.index("--compaction-budget") + 1] == "16384"


def test_the_driver_refuses_a_withdrawn_manifest(tmp_path, monkeypatch):
    d = driver()
    monkeypatch.setattr(d, "key", lambda: pytest.fail("a withdrawn block must not read the key"))
    monkeypatch.setattr(d.subprocess, "Popen", lambda *a, **kw: pytest.fail("a withdrawn block must not launch"))
    out = tmp_path / "blocks.jsonl"
    mf = tmp_path / "m.json"
    mf.write_text(json.dumps({**MANIFEST, "withdrawn": "not to be played"}))
    monkeypatch.setattr(d, "WT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["driver.py", str(mf), str(out)])
    with pytest.raises(SystemExit):
        d.main()
    log = json.loads(out.read_text())
    assert log["started"] is False and "withdrawn" in log["why"]


def test_the_drivers_jev_count_skips_rows_from_before_the_launch(tmp_path):
    d = driver()
    t = tmp_path / "a" / "t" / "r0" / "run" / "x"
    t.mkdir(parents=True)
    (t / "trace.jsonl").write_text(
        json.dumps({"ts": 100.0, "kind": "request", "usage": {"input_tokens": 7}}) + "\n" +
        json.dumps({"ts": 300.0, "kind": "request", "usage": {"input_tokens": 5}}) + "\n")
    assert d.jev_tokens(tmp_path, [{"arms": ["a"], "tasks": ["t"], "start": 0, "repeats": 1}], 200.0) == 5


# ---- dry runs and the full run's manifest

def test_run_py_dry_run_lists_the_jobs_and_plays_none(tmp_path, monkeypatch, capsys):
    exp = suite(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "run_one", lambda *a, **kw: pytest.fail("a dry run must not play a run"))
    monkeypatch.setattr(sys, "argv", ["run.py", "fake", "--arms", "a", "--repeats", "2", "--tag", "x",
                                      "--no-task-env", "--note", "test", "--dry-run"])
    runner.main()
    assert capsys.readouterr().out.splitlines() == ["x:fake/a/t1/r0", "x:fake/a/t2/r0", "x:fake/a/t1/r1",
                                                    "x:fake/a/t2/r1", "4 runs"]
    assert not (exp / "runs-x").exists() and not (tmp_path / "work").exists()


def test_run_py_dry_run_with_skip_done_leaves_out_finished_runs(tmp_path, monkeypatch, capsys):
    exp = suite(tmp_path, monkeypatch)
    f = exp / "runs-x" / "fake" / "a" / "records.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text("\n".join(json.dumps(r) for r in (
        {"rep": 0, "task": "t1"}, {"rep": 0, "task": "t2", "rerun": True}, {"rep": 1, "task": "t1", "stopped": True})))
    monkeypatch.setattr(sys, "argv", ["run.py", "fake", "--arms", "a", "--repeats", "2", "--tag", "x",
                                      "--no-task-env", "--note", "test", "--skip-done", "--dry-run"])
    runner.main()
    # a jev-error run and a stopped run play again
    assert capsys.readouterr().out.splitlines() == ["x:fake/a/t2/r0", "x:fake/a/t1/r1", "x:fake/a/t2/r1", "3 runs"]


def test_the_driver_passes_skip_done_only_when_the_manifest_sets_it():
    d = driver()
    m = MANIFEST
    assert "--skip-done" not in d.cmd(m, m["cells"][0])
    assert "--skip-done" in d.cmd({**m, "skip_done": True}, m["cells"][0])


def test_the_drivers_dry_run_reads_no_key_and_asks_run_py_for_a_dry_run(tmp_path, monkeypatch):
    d = driver()
    monkeypatch.setattr(d, "key", lambda: pytest.fail("a dry run must not read the key"))
    monkeypatch.setattr(d.subprocess, "Popen", lambda *a, **kw: pytest.fail("a dry run must not launch"))
    got = []
    monkeypatch.setattr(d.subprocess, "run", lambda c, **kw: got.append(c))
    mf = tmp_path / "m.json"
    mf.write_text(json.dumps({**MANIFEST, "skip_done": True}))
    monkeypatch.setattr(d, "WT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["driver.py", str(mf), "--dry-run"])
    d.main()
    assert len(got) == 1 and got[0][-1] == "--dry-run" and "--skip-done" in got[0]


def test_the_32k_manifest_plays_every_version_once():
    mp = ROOT / "experiments" / "main" / "manifest_gemma4-31b-32k.json"
    if not mp.exists():
        pytest.skip("the manifests come with the data build")
    m = json.loads(mp.read_text())
    full = json.loads((ROOT / "experiments" / "main" / "tasks.json").read_text())
    confs = ["plain", "relevance", "injection", "relevance-injection"]
    assert set(runner.tomllib.loads((ROOT / "experiments" / "main" / "test.toml").read_text())["arms"]) == set(confs)
    assert (m["suite"], m["tag"], m["compaction_budget"], m["max_turns"], m["skip_done"]) == \
        ("main", "gemma4-31b-32k", 32768, 40, True)
    b16 = json.loads((ROOT / "experiments" / "main" / "manifest_gemma4-31b-16k.json").read_text())
    for k in ("model", "provider", "max_output", "sampling", "context_window_tokens"):
        assert m[k] == b16[k], k
    played = [(a, t, r) for c in m["cells"] for t in c["tasks"] for a in c["arms"]
              for r in range(c["start"], c["start"] + c["repeats"])]
    assert len(played) == len(set(played)) == m["runs"] == 600 and m["reused"] == []
    assert set(played) == {(a, t["id"], 0) for t in full["tasks"] for a in confs}
    assert all(c["arms"] == confs and c["workers"] == 12 for c in m["cells"])
    # test beds take turns in the order, so a partial stop keeps all three
    first = [t.split("-")[0] for t in m["cells"][0]["tasks"][:12]]
    assert {"cb", "lbf", "dfq"} <= set(first)
    assert m["stop_at"]["openrouter_usd"] <= m["budget_usd"]
