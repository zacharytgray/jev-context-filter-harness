"""run one test's tasks through the harness, for one or more arms, several times each.

    uv run python experiments/run.py gate --arms on,off --repeats 10
    uv run python experiments/run.py filter --arms on --repeats 1 --tasks long-log

each run gets a fresh copy of the task repo, then the task's check grades it (a benchmark check writes a score).
runs land in experiments/runs/<test>/<arm>/<task>/r<k>/, one record per run in
experiments/runs/<test>/<arm>/records.jsonl. arms interleave so drift on the model server hits both.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from metrics import run_metrics

EXP = Path(__file__).resolve().parent
ROOT = EXP.parent
RUN_TIMEOUT = 3600  # 1800 cut off long core runs
JEV_ERROR_EXIT = 3  # s1harness.cli's exit code for a run stopped by a jev error
# agent workdirs live outside the repo, so bash cannot climb into experiments/ and read the checks
# S1H_WORK_ROOT moves them. the default keeps the recorded runs' paths on macos
WORK_ROOT = Path(os.environ.get("S1H_WORK_ROOT")
                 or ("/private/tmp/s1h-eval" if sys.platform == "darwin" else "/tmp/s1h-eval"))
# the agent's shell gets a python with pytest first on its PATH, so a request to keep the tests passing
# can be checked. without it, pytest is missing and pip cannot fetch it. it lives outside the repo, so a path
# in pytest output does not lead to the task sources
TASK_ENV = Path.home() / ".cache" / "s1h-taskenv"
TASK_PYTEST = "pytest==9.1.1"  # the harness venv's version, which the checks use
CLI = [sys.executable, "-m", "s1harness.cli"]
# a sigterm from a driver stops the sweep: live runs get it too, so they write their records, and no new run starts
STOP = threading.Event()
LIVE: set[subprocess.Popen] = set()
LIVE_LOCK = threading.Lock()


def on_term(*_):
    STOP.set()
    with LIVE_LOCK:
        for p in LIVE:
            p.terminate()


def task_env() -> Path:
    py = TASK_ENV / "bin" / "python3"
    if not py.exists():
        subprocess.run(["uv", "venv", "--python", "3.14", str(TASK_ENV)], check=True)
        subprocess.run(["uv", "pip", "install", "--python", str(py), TASK_PYTEST], check=True)
    return TASK_ENV / "bin"


def load_test(test: str) -> tuple[dict, dict[str, dict]]:
    tdir = EXP / test
    spec = tomllib.loads((tdir / "test.toml").read_text())
    tasks = {}
    for d in sorted((tdir / "tasks").iterdir()):
        if (d / "task.toml").exists():
            tasks[d.name] = {"dir": d, **tomllib.loads((d / "task.toml").read_text())}
    return spec, tasks


def prepare(task: dict, work: Path, setup: bool = True):
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(task["dir"] / "repo", work, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
    git = ["git", "-c", "user.email=eval@local", "-c", "user.name=eval"]
    subprocess.run([*git, "init", "-q"], cwd=work, check=True)
    subprocess.run([*git, "add", "-A"], cwd=work, check=True)
    subprocess.run([*git, "commit", "-qm", "initial"], cwd=work, check=True)
    if setup and (task["dir"] / "setup.sh").exists():  # state after the first commit, like uncommitted edits
        subprocess.run(["bash", str(task["dir"] / "setup.sh")], cwd=work, check=True)


def check(task: dict, work: Path, run_dir: Path | None = None) -> dict:
    """check.py <workdir> [<run_dir>]. the run dir lets a check read what entered context. older checks ignore it"""
    r = subprocess.run([sys.executable, str(task["dir"] / "check.py"), str(work), *([str(run_dir)] if run_dir else [])],
                       capture_output=True, text=True, timeout=300, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        out = {"score": None, "detail": f"check crashed: {r.stderr[-500:]}"}
    return out


def runs_root(llm: dict) -> Path:
    # untagged runs stay in runs/, a tagged model gets its own folder so the two never mix
    return EXP / ("runs" if not llm.get("tag") else f"runs-{llm['tag']}")


def code_version(test: str) -> dict:
    """the commit a sweep ran on, and whether the tree had uncommitted changes in the harness, the lever files, the
    runner, the metrics, the benchmark adapters (their check and scorers are copied into the tasks), or the test's
    own folder"""
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    paths = ["s1harness", "levers", "experiments/run.py", "experiments/metrics.py", "experiments/bench",
             f"experiments/{test}"]
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", *paths],
                                cwd=ROOT, capture_output=True, text=True).stdout.strip())
    return {"commit": head, "dirty": dirty}


def run_one(test: str, arm: str, arm_spec: dict, name: str, task: dict, rep: int, llm: dict,
            label: dict | None = None) -> dict:
    base = runs_root(llm) / test / arm / name / f"r{rep}"
    if base.exists():
        shutil.rmtree(base)
    work = WORK_ROOT / (llm.get("tag") or "") / test / arm / name / f"r{rep}"
    runs = base / "run"
    runs.mkdir(parents=True)
    prepare(task, work)
    (base / "request.txt").write_text(task["request"])
    cmd = [*CLI, f"@{base / 'request.txt'}", "--workdir", str(work),
           "--levers", str(ROOT / arm_spec["levers"]), "--runs-dir", str(runs)]
    for s in arm_spec.get("set", []):
        cmd += ["--set", s]
    for flag in ("base_url", "model", "api_key_env", "provider", "max_context", "context_window", "reasoning_effort",
                 "max_output", "temperature", "top_p", "top_k", "max_turns", "compaction_budget"):
        if llm.get(flag):
            cmd += ["--" + flag.replace("_", "-"), str(llm[flag])]
    # vllm's sampling, passed when set, a 0 included
    for flag in ("min_p", "presence_penalty", "chat_template_kwargs"):
        if llm.get(flag) is not None:
            v = llm[flag]
            cmd += ["--" + flag.replace("_", "-"), json.dumps(v) if isinstance(v, dict) else str(v)]
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.monotonic()
    try:
        # no package installs from agent bash: they would land in the harness venv
        env = {**os.environ, "PIP_NO_INDEX": "1", "UV_OFFLINE": "1"}
        if llm.get("task_env"):
            env["PATH"] = f"{task_env()}:{env.get('PATH', '')}"
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT,
                             stdin=subprocess.DEVNULL, env=env)
        with LIVE_LOCK:
            LIVE.add(p)
            if STOP.is_set():  # the signal came while this run was starting
                p.terminate()
        try:
            _, stderr = p.communicate(timeout=RUN_TIMEOUT)
            rc, err = p.returncode, stderr[-2000:]
        except subprocess.TimeoutExpired:
            # sigterm first, so the run writes its records
            p.terminate()
            try:
                p.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate()
            rc, err = -1, "timeout"
        finally:
            with LIVE_LOCK:
                LIVE.discard(p)
    except OSError as e:
        rc, err = -1, repr(e)
    wall = time.monotonic() - t0
    (base / "stderr.txt").write_text(err)
    dirs = sorted(runs.iterdir())
    run_dir = dirs[-1] if dirs else None
    rec = {"test": test, "arm": arm, "task": name, "rep": rep, "model": llm.get("model"),
           "started": started, **(label or {}),
           "wall_s": round(wall, 2), "exit": rc,
           "run_dir": str(run_dir.relative_to(ROOT)) if run_dir else None}
    if STOP.is_set():  # cut short by the sweep's sigterm, so --skip-done runs it again
        rec["stopped"] = True
    rec.update(check(task, work, run_dir))
    diff = subprocess.run("git add -A && git -c core.pager=cat diff --cached --stat && git diff --cached",
                          shell=True, cwd=work, capture_output=True, text=True).stdout
    (base / "work.diff").write_text(diff[:200000])
    if run_dir:
        rec.update(run_metrics(run_dir, task))
    if rc == JEV_ERROR_EXIT or rec.get("rerun"):
        # a jev error: the run is rerun and replaced, not a crash of the harness
        rec["rerun"] = True
    elif rc != 0:
        # a crashed run writes its own end (overflow, error) to result.json. keep it when there
        if not (run_dir and (run_dir / "result.json").exists()):
            rec["end"] = "crash"
        rec["crash"] = err.strip().splitlines()[-1] if err.strip() else f"exit {rc}"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    ap.add_argument("--arms", required=True, help="comma list, from the test's test.toml")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--start", type=int, default=0, help="first repeat index, to extend a sweep")
    ap.add_argument("--tasks", default=None, help="comma list, default all")
    ap.add_argument("--tag", default=None, help="name for a hosted model's runs folder, e.g. deepseek")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--provider", default=None, help="pin one openrouter provider, no fallbacks")
    ap.add_argument("--max-context", type=int, default=None, help="fail calls past this many prompt tokens")
    ap.add_argument("--context-window", type=int, default=None,
                    help="the model's window in tokens, for compaction. default: the cli's, 262144")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--max-output", type=int, default=None, help="max_tokens per llm call. novita needs it")
    ap.add_argument("--workers", type=int, default=1, help="runs at once. keep 1 for a local model on one gpu")
    ap.add_argument("--skip-done", action="store_true", help="skip runs that already have a record, to finish a sweep that stopped")
    ap.add_argument("--dry-run", action="store_true", help="list the runs this sweep would play, in order, and stop")
    ap.add_argument("--no-task-env", action="store_true", help="leave pytest off the agent's PATH")
    ap.add_argument("--temperature", type=float, default=None, help="sampling for every arm")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None, help="vllm's, sent only when given")
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--chat-template-kwargs", type=json.loads, default=None, metavar="JSON",
                    help='e.g. \'{"enable_thinking": false}\', sent only when given')
    ap.add_argument("--max-turns", type=int, default=None,
                    help="the harness's turn limit per run. default: the test.toml's max_turns, else the harness's 40")
    ap.add_argument("--compaction-budget", type=int, default=None,
                    help="compaction triggers past this many prompt tokens. default: the cli's, 32768")
    ap.add_argument("--note", required=True,
                    help="a short note on this sweep, e.g. 'Gemma 4 31B, 32K compaction budget'. every record keeps it")
    a = ap.parse_args()
    llm = {"tag": a.tag, "model": a.model, "base_url": a.base_url, "api_key_env": a.api_key_env, "provider": a.provider,
           "max_context": a.max_context, "context_window": a.context_window,
           "reasoning_effort": a.reasoning_effort, "max_output": a.max_output,
           "task_env": not a.no_task_env, "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k,
           "min_p": a.min_p, "presence_penalty": a.presence_penalty, "chat_template_kwargs": a.chat_template_kwargs,
           "compaction_budget": a.compaction_budget}

    spec, tasks = load_test(a.test)
    # the command line wins over the test.toml's max_turns. with neither the harness's default holds
    llm["max_turns"] = a.max_turns or spec.get("max_turns")
    label = {**code_version(a.test), "note": a.note, "task_env": llm["task_env"],
             "sampling": {k: llm[k] for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                                              "chat_template_kwargs") if llm[k] is not None},
             **({"max_turns": llm["max_turns"]} if llm["max_turns"] else {}),
             **({"compaction_budget": llm["compaction_budget"]} if llm["compaction_budget"] else {})}
    if label["dirty"]:
        print("warning: uncommitted changes in the harness or the tasks. commit first so runs map to a version",
              flush=True)
    arms = a.arms.split(",")
    names = a.tasks.split(",") if a.tasks else list(tasks)
    jobs = [(rep, name, arm) for rep in range(a.start, a.start + a.repeats) for name in names
            for arm in (arms if rep % 2 == 0 else arms[::-1])]  # alternate which arm goes first
    if a.skip_done:
        done = set()
        for arm in arms:
            f = runs_root(llm) / a.test / arm / "records.jsonl"
            if f.exists():
                # a run stopped by a jev error is not done: it is rerun and replaced
                done |= {(r["rep"], r["task"], arm) for r in map(json.loads, f.read_text().splitlines())
                         if not r.get("rerun") and not r.get("stopped")}
        jobs = [j for j in jobs if j not in done]
    if a.dry_run:  # nothing is copied, called or written
        for rep, name, arm in jobs:
            print(f"{a.tag + ':' if a.tag else ''}{a.test}/{arm}/{name}/r{rep}", flush=True)
        print(f"{len(jobs)} runs", flush=True)
        return
    lock = threading.Lock()

    def one(job):
        rep, name, arm = job
        if STOP.is_set():
            return
        try:
            rec = run_one(a.test, arm, spec["arms"][arm], name, tasks[name], rep, llm, label)
        except Exception as e:  # one broken run must not stop the sweep
            print(f"{a.test}/{arm}/{name}/r{rep}: runner error {e!r}", flush=True)
            return
        with lock:
            with open(runs_root(llm) / a.test / arm / "records.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            graded = f"score={rec.get('score')}" if "score" in rec else f"pass={rec.get('pass')}"
            print(f"{a.tag + ':' if a.tag else ''}{a.test}/{arm}/{name}/r{rep}: {graded} "
                  f"end={rec.get('end')}{' rerun' if rec.get('rerun') else ''} turns={rec.get('turns')} "
                  f"wall={rec['wall_s']}s "
                  f"{rec.get('detail', '')[:80]}", flush=True)

    signal.signal(signal.SIGTERM, on_term)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, jobs))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
