"""one frozen paid block. reads its manifest, checks the key and the budget the manifest records, launches one run.py
per cell at once, and watches spend while they run: the key's usage for openrouter, the runs' own traces for jev. past
the block's stop point on either, every run.py gets sigterm (it passes it on, so the runs keep their records). logs to
the given jsonl. --dry-run in place of the log lists each cell's runs through run.py --dry-run, with no key read, no
call and no log.

a manifest with endpoint {"kind": "local", "base_url": ..., "api_key_env": null or a name} and no provider runs on a
self-hosted server: no openrouter key, budget or usd stop (openrouter_usd stays 0 in the log), the jev stop as before,
and the launch is refused unless GET <base_url>/models lists the manifest's model.

    driver.py experiments/main/manifest_<tag>.json experiments/main/blocks_<tag>.jsonl
    driver.py experiments/main/manifest_<tag>.json --dry-run"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parent
# this checkout's own venv (uv sync makes it), so run.py and the harness are this repo's
PY = str(WT / ".venv" / "bin" / "python")
POLL = 30


def key() -> dict:
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)["data"]
    return {"usage": d["usage"], "limit_remaining": d["limit_remaining"]}


def local(m: dict) -> bool:
    return (m.get("endpoint") or {}).get("kind") == "local"


def models(base_url: str, key_env: str | None) -> list[str]:
    """the model ids a server lists at /models"""
    headers = {"Authorization": f"Bearer {os.environ[key_env]}"} if key_env else {}
    req = urllib.request.Request(base_url.rstrip("/") + "/models", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return [d["id"] for d in json.load(r)["data"]]


def jev_tokens(root: Path, cells: list[dict], since: float) -> int:
    """jev input tokens in the traces of this block's run dirs, from rows written after the launch"""
    n = 0
    for c in cells:
        for arm in c["arms"]:
            for task in c["tasks"]:
                for rep in range(c["start"], c["start"] + c["repeats"]):
                    for t in (root / arm / task / f"r{rep}").glob("**/trace.jsonl"):
                        if t.stat().st_mtime < since:
                            continue
                        for line in t.open():
                            r = json.loads(line)
                            if r.get("kind") == "request" and r.get("ts", since) >= since:
                                n += ((r.get("usage") or {}).get("input_tokens") or 0)
    return n


def cmd(m: dict, c: dict) -> list[str]:
    s = m["sampling"]
    if local(m):  # a self-hosted server: its own url, a key only if it needs one, no provider, its own window
        e = m["endpoint"]
        llm = ["--base-url", e["base_url"], *(["--api-key-env", e["api_key_env"]] if e.get("api_key_env") else []),
               *(["--max-output", str(m["max_output"])] if m.get("max_output") else []),
               "--context-window", str(m["context_window_tokens"])]
    else:
        llm = ["--base-url", "https://openrouter.ai/api/v1", "--api-key-env", "OPENROUTER_API_KEY",
               "--provider", m["provider"], "--max-output", str(m["max_output"])]
    return [PY, "experiments/run.py", m["suite"], "--arms", ",".join(c["arms"]), "--tasks", ",".join(c["tasks"]),
            "--start", str(c["start"]), "--repeats", str(c["repeats"]), "--workers", str(c["workers"]),
            "--tag", m["tag"], "--model", m["model"], *llm,
            "--temperature", str(s["temperature"]),
            # top_p and top_k only when the manifest sets them, so a temperature-only manifest keeps the server's
            # defaults for the rest
            *(["--top-p", str(s["top_p"])] if "top_p" in s else []),
            *(["--top-k", str(s["top_k"])] if "top_k" in s else []),
            # vllm's sampling, only when the manifest sets it
            *(["--min-p", str(s["min_p"])] if "min_p" in s else []),
            *(["--presence-penalty", str(s["presence_penalty"])] if "presence_penalty" in s else []),
            *(["--chat-template-kwargs", json.dumps(s["chat_template_kwargs"])] if "chat_template_kwargs" in s else []),
            # a reasoning model's effort (the request's reasoning.effort), only when the manifest sets it
            *(["--reasoning-effort", m["reasoning_effort"]] if m.get("reasoning_effort") else []),
            # a manifest's max_turns is the run's turn limit
            *(["--max-turns", str(m["max_turns"])] if m.get("max_turns") else []),
            # a manifest's compaction_budget is the run's own; without one the harness's 32,768 holds
            *(["--compaction-budget", str(m["compaction_budget"])] if m.get("compaction_budget") else []),
            # a manifest's skip_done makes a relaunch go on from the records already written
            *(["--skip-done"] if m.get("skip_done") else []),
            "--note", f"{m['note']} ({m['manifest']})"]


def main():
    mpath = Path(sys.argv[1]).resolve()
    m = json.loads(mpath.read_text())
    m["manifest"] = str(mpath.relative_to(WT))
    if sys.argv[2] == "--dry-run":
        for c in m["cells"]:
            subprocess.run([*cmd(m, c), "--dry-run"], cwd=WT, check=True)
        print(f"{len(m.get('reused') or [])} runs reused, not played", flush=True)
        return
    out = Path(sys.argv[2]).resolve()

    def log(entry):
        entry["at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with out.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    if m.get("withdrawn"):  # a withdrawn block never launches
        log({"block": m["block"], "started": False, "why": "the manifest is withdrawn: " + m["withdrawn"]})
        sys.exit(1)
    stop = m["stop_at"]
    if local(m):  # no key to read: the server must answer with the manifest's model, or nothing launches
        e = m["endpoint"]
        try:
            served = models(e["base_url"], e.get("api_key_env"))
        except Exception as err:
            log({"block": m["block"], "started": False, "why": f"the endpoint did not answer: {err!r}"})
            sys.exit(1)
        if m["model"] not in served:
            log({"block": m["block"], "started": False, "served": served,
                 "why": "the endpoint does not serve the manifest's model"})
            sys.exit(1)
        before = {"endpoint": e["base_url"], "served": served}
    else:
        before = key()
        if before["limit_remaining"] < stop["openrouter_usd"] * 1.1:
            log({"block": m["block"], "started": False, **before, "why": "key limit below the block's stop point"})
            sys.exit(1)
        if stop["openrouter_usd"] > m["budget_usd"]:
            log({"block": m["block"], "started": False, "why": "the stop point is over the approved budget"})
            sys.exit(1)
    since = time.time()
    log({"block": m["block"], "started": True, **before})
    root = WT / "experiments" / f"runs-{m['tag']}" / m["suite"]
    procs = [subprocess.Popen(cmd(m, c), cwd=WT) for c in m["cells"]]
    stopped = None
    while any(p.poll() is None for p in procs):
        time.sleep(POLL)
        try:
            spent = 0.0 if local(m) else key()["usage"] - before["usage"]
        except Exception as e:  # a failed read skips one check, never the stop
            log({"block": m["block"], "key_read_failed": repr(e)})
            continue
        jev = jev_tokens(root, m["cells"], since)
        log({"block": m["block"], "openrouter_usd": round(spent, 4), "jev_input_tokens": jev})
        # a local block has no usd stop: it may leave stop_at's openrouter_usd out
        if stopped is None and ((not local(m) and spent > stop["openrouter_usd"]) or jev > stop["jev_input_tokens"]):
            stopped = {"openrouter_usd": round(spent, 4), "jev_input_tokens": jev}
            log({"block": m["block"], "stopping": "past the stop point", **stopped})
            for p in procs:
                if p.poll() is None:
                    p.send_signal(signal.SIGTERM)
    rcs = [p.wait() for p in procs]
    if local(m):
        log({"block": m["block"], "done": True, "exits": rcs, "openrouter_usd": 0.0,
             "jev_input_tokens": jev_tokens(root, m["cells"], since), "stopped": stopped})
        return
    time.sleep(90)  # let the key's usage catch up
    after = key()
    log({"block": m["block"], "done": True, "exits": rcs, **after, "openrouter_usd": round(after["usage"] - before["usage"], 4),
         "jev_input_tokens": jev_tokens(root, m["cells"], since), "stopped": stopped})


if __name__ == "__main__":
    main()
