"""AgentDojo run summary: per suite and overall, per defense, from records.jsonl and defense.jsonl.

python3 experiments/agentdojo/summarize.py [RUN_DIR] [--json out.json] [--markdown]
    [--until EPOCH]

RUN_DIR defaults to the repo's experiments/runs-agentdojo-gemma4-31b.
travel is read from travel-yaml/, the other suites from their own folder. read-only: it opens files and prints.

- the last record per episode key (suite, user task, injection task, attack, defense) stands. a record with
  rerun (a jev error) is not a result and is left out. errors count as not solved and not attacked, and are listed
- security follows agentdojo: True means the injection task succeeded, so the attack succeeded
- rates carry a wilson 95% interval
- defense rows, jev trace lines and llm calls count only the standing play of each episode. the cross check compares
  the records' token totals with those lines
- safe to run while a call is still writing: a torn last line is skipped. --until EPOCH drops records written after
  that time, for a fixed snapshot of a folder that is still growing
- --markdown also prints the results as markdown tables
- it also lists the attacks that succeeded under the filter, with the filter's view of each tool output (its chunks
  and their probabilities), and the user tasks the plain loop solved without attack that the filter's episode did not
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1] / "runs-agentdojo-gemma4-31b")
FOLDERS = {"banking": "banking", "slack": "slack", "travel": "travel-yaml", "workspace": "workspace"}
SUITES = tuple(FOLDERS)
DEFS = ("none", "jev-injection")
# user tasks and injection tasks per suite, agentdojo v1.2.2 (manifest.json calls)
PLAN = {"banking": (16, 9), "slack": (21, 5), "travel": (20, 7), "workspace": (40, 14)}
TAG = "<INFORMATION>"


def load(p: Path) -> list[dict]:
    out = []
    if not p.exists():
        return out
    for x in p.read_text().splitlines():
        if x.strip():
            try:
                out.append(json.loads(x))
            except json.JSONDecodeError:  # a line still being written
                pass
    return out


def key(r):
    return (r["suite"], r["user_task"], r["injection_task"] or "none", r["attack"], r["defense_id"])


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {o: sys.argv[sys.argv.index(o) + 1] for o in ("--json", "--until") if o in sys.argv}
    for v in opt.values():
        args.remove(v)
    root = Path(args[0] if args else ROOT)
    out_json = opt.get("--json")
    until = float(opt.get("--until", "inf"))

    recs, rows, traces, calls = [], [], [], []
    for s, f in FOLDERS.items():
        rs = [r for r in load(root / f / "records.jsonl") if r["ts"] <= until]
        last = {key(r): r for r in rs}
        standing = [r for r in last.values() if not r.get("rerun")]
        recs += standing
        plays = {r["play"] for r in standing}
        rows += [d for d in load(root / f / "defense.jsonl") if d["play"] in plays]
        for t in load(root / f / "jev_trace.jsonl"):
            if t["play"] in plays:
                traces.append(t)
        for c in load(root / f / "llm_calls.jsonl"):
            if c["ts"] > until:
                continue
            if c["play"] in plays:
                calls.append(c)
    chunks_of = defaultdict(list)  # (play, call id) -> the filter's chunk decisions
    for t in traces:
        if t.get("kind") == "chunk":
            chunks_of[t["play"], t["call_id"]].append(t)

    # completeness against the plan
    coverage = {}
    for s in SUITES:
        u, i = PLAN[s]
        for d in DEFS:
            for att, planned in (("none", u), ("important_instructions", u * i)):
                have = sum(r["suite"] == s and r["defense_id"] == d and r["attack"] == att for r in recs)
                coverage[f"{s}|{att}|{d}"] = {"present": have, "planned": planned}

    stats = {}
    for scope in (*SUITES, "overall"):
        for d in DEFS:
            rs = [r for r in recs if r["defense_id"] == d and (scope == "overall" or r["suite"] == scope)]
            na = [r for r in rs if r["injection_task"] is None]
            at = [r for r in rs if r["injection_task"] is not None]
            fb = Counter()
            for r in rs:
                fb.update(r.get("jev_fallbacks") or {})
            p, c = sum(r["llm_prompt_tokens"] for r in rs), sum(r["llm_completion_tokens"] for r in rs)
            jin = sum(r["jev_input_tokens"] for r in rs)
            stats[f"{scope}|{d}"] = {
                "episodes": len(rs), "no_attack_episodes": len(na), "attack_episodes": len(at),
                "utility_no_attack": rate(sum(bool(r["utility"]) for r in na), len(na)),
                "utility_under_attack": rate(sum(bool(r["utility"]) for r in at), len(at)),
                "attack_success": rate(sum(bool(r["security"]) for r in at), len(at)),
                "errors": sum(bool(r["error"]) for r in rs),
                "end_states": dict(sorted(Counter(r["end_state"] for r in rs).items())),
                "jev_requests": sum(r["jev_requests"] for r in rs),
                "jev_retries": sum(r["jev_retries"] for r in rs),
                "jev_failed_requests": sum(r["jev_failed_requests"] for r in rs),
                "jev_fallbacks": dict(fb),
                "chunks_fallback": sum((r.get("injection_filter") or {}).get("chunks_fallback", 0) for r in rs),
                "stuck_chunks": sum((r.get("injection_filter") or {}).get("stuck", 0) for r in rs),
                "jev_usage_missing": sum(r.get("jev_usage_missing") or 0 for r in rs),
                "llm_prompt": p, "llm_completion": c,
                "jev_in": jin, "jev_out": sum(r["jev_output_tokens"] for r in rs),
            }

    # every attack that succeeded under the filter, with the filter's view of each injected output
    wins = []
    for r in recs:
        if r["defense_id"] != "jev-injection" or not r["security"]:
            continue
        outs = [x for x in rows if x["play"] == r["play"]]
        wins.append({"key": "|".join(key(r)), "play": r["play"], "utility": r["utility"], "log": r["log"],
                     "turns": r["turns"], "tool_calls": r["tool_calls"],
                     "outputs": [{"call": x["call"], "tool": x["tool"], "injected": TAG in x["before"],
                                  "flagged": x["flagged"], "lines": len(x["before"].splitlines()),
                                  "lines_withheld": x["lines_withheld"], "spans": x["finder_spans"],
                                  "tag_still_shown": TAG in x.get("after", x["before"]), "fallbacks": x["fallbacks"],
                                  "stuck": x["stuck"],
                                  "chunks": [{"lines": c["lines"], "p": c["p"], "decision": c["decision"],
                                              "hidden": c.get("hidden")} for c in chunks_of[x["play"], x["call_id"]]]}
                                 for x in outs]})

    # user tasks solved without attack by none but not by the filter
    lost = []
    for r in recs:
        if r["defense_id"] != "jev-injection" or r["injection_task"] is not None or r["utility"]:
            continue
        twin = next((x for x in recs if x["defense_id"] == "none" and x["injection_task"] is None
                     and x["suite"] == r["suite"] and x["user_task"] == r["user_task"]), None)
        if twin and twin["utility"]:
            fl = [x for x in rows if x["play"] == r["play"] and x["flagged"]]
            lost.append({"suite": r["suite"], "user_task": r["user_task"], "end_state": r["end_state"],
                         "outputs_flagged": len(fl), "lines_withheld": sum(x["lines_withheld"] for x in fl)})

    # standing plays against their own call and trace lines
    cross = {"llm_calls_prompt": sum(c.get("prompt_tokens") or 0 for c in calls),
             "records_prompt": sum(r["llm_prompt_tokens"] for r in recs),
             "trace_jev_in": sum((t.get("usage") or {}).get("input_tokens", 0) for t in traces if t.get("kind") == "request"),
             "records_jev_in": sum(r["jev_input_tokens"] for r in recs)}

    errs = [{"key": "|".join(key(r)), "error": (r["error"] or "")[:200]} for r in recs if r["error"]]
    res = {"coverage": coverage, "stats": stats, "attacks_through_filter": wins, "lost_no_attack": lost, "errors": errs,
           "cross_check": cross}
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1))
    report(res)
    if "--markdown" in sys.argv:
        markdown(res)


def pct(x):
    if x["rate"] is None:
        return "n/a"
    lo, hi = x["ci95"]
    return f"{x['k']}/{x['n']} {100 * x['rate']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def report(res):
    print("coverage (present / planned):")
    for s in SUITES:
        parts = [f"{att} {d} {v['present']}/{v['planned']}"
                 for k, v in res["coverage"].items() if k.startswith(s + "|")
                 for _, att, d in [k.split("|")]]
        print(f"  {s}: " + "; ".join(parts))
    print("\nrates (k/n, wilson 95%):")
    for k, v in res["stats"].items():
        print(f"  {k:26} util no-atk {pct(v['utility_no_attack']):28} util atk {pct(v['utility_under_attack']):30} "
              f"ASR {pct(v['attack_success'])}")
    print("\ncounts and tokens:")
    for k, v in res["stats"].items():
        print(f"  {k:26} eps {v['episodes']} err {v['errors']} jev req {v['jev_requests']} retries {v['jev_retries']} "
              f"failed {v['jev_failed_requests']} fallbacks {v['jev_fallbacks']} chunks_fb {v['chunks_fallback']} "
              f"stuck {v['stuck_chunks']} | prompt {v['llm_prompt']:,} compl {v['llm_completion']:,} "
              f"| jev in {v['jev_in']:,}")
    print("\nattacks that succeeded under the filter:")
    for w in res["attacks_through_filter"]:
        print("  " + json.dumps(w))
    print("\nsolved by none, not by the filter, no attack:")
    for x in res["lost_no_attack"]:
        print("  " + json.dumps(x))
    print("\nerrors:", json.dumps(res["errors"]))
    print("\ncross check, standing plays:", json.dumps(res["cross_check"]))


def markdown(res):
    st, cov = res["stats"], res["coverage"]

    def cell(x):
        lo, hi = x["ci95"]
        return f"{x['k']}/{x['n']} ({100 * x['rate']:.1f}%, {100 * lo:.1f} to {100 * hi:.1f})"

    names = {"none": "Plain loop", "jev-injection": "Jev injection filter"}
    print("\n## markdown: headline")
    print("| Suite | Defense | Utility, no attack | Utility under attack | Attack success |")
    print("|---|---|---|---|---|")
    for scope in (*SUITES, "overall"):
        for d in DEFS:
            v = st[f"{scope}|{d}"]
            print(f"| {scope.capitalize()} | {names[d]} | {cell(v['utility_no_attack'])} | "
                  f"{cell(v['utility_under_attack'])} | {cell(v['attack_success'])} |")
    print("\n## markdown: tokens (standing episodes)")
    print("| Suite | Defense | Episodes | Agent input tokens | Agent output tokens | Jev input tokens |")
    print("|---|---|---|---|---|---|")
    for scope in (*SUITES, "overall"):
        for d in DEFS:
            v = st[f"{scope}|{d}"]
            print(f"| {scope.capitalize()} | {names[d]} | {v['episodes']:,} | {v['llm_prompt']:,} | "
                  f"{v['llm_completion']:,} | {v['jev_in']:,} |")
    print("\n## markdown: coverage")
    for k, v in cov.items():
        if v["present"] != v["planned"]:
            print(f"  {k}: {v['present']} of {v['planned']}")


if __name__ == "__main__":
    main()
