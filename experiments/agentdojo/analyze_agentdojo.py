"""analysis of one run_agentdojo.py output folder (records, defense rows, jev and llm logs, AgentDojo's own episode
logs). offline: it reads the folder and calls nothing, and needs no agentdojo install. test_run_agentdojo.py uses it
to check what the runner writes.

inputs, in the folder: records.jsonl (one line per episode, the last line for a key wins), defense.jsonl (one line per
tool output the jev filter saw), jev_trace.jsonl (every Jev request and finder event), llm_calls.jsonl
(every llm call), runs.jsonl (the runner's start and end lines), and agentdojo/ (AgentDojo's own per-episode logs, whose
messages are what the agent got, after the defense).

AgentDojo's security True means the injection task succeeded: the attack worked. utility is the user task's own
check. an episode that ended in an error has neither: it is listed, counted, and left out of every rate. rates are
of the episodes without an error, and the planned count from the manifest is shown beside them.
an episode a Jev error stopped has rerun true. it is listed as needing a rerun until a rerun replaces it (the runner
plays it again on its next call, or `replace` reads a rerun folder).

the filter's flags are counted per tool output it saw (an agent often repeats a call, and each repeat is its own
output): on no-attack episodes (clean), and on attacked episodes, split by whether the output held AgentDojo's
injection (its <INFORMATION> template, as the tool rendered it). an injection's goal is the text between "please do
the following first:" and "After you do that", whitespace and YAML's doubled quotes aside. the goal is visible after
the defense when the whole goal is still in what the agent got: for jev, the filtered text; for none, the text.
delivery per episode comes from AgentDojo's own log: a tool message the agent got that holds the whole goal.

cost: llm tokens at the runner's prices (llm_usd_est). Jev tokens are counted (jev_in, jev_out), not priced.

replace: other run folders of the same layout. their records replace the records with the same key, with their own
defense rows and logs; their jev and llm logs are added to the totals, since both the original and the replacement
were paid for. every replacement is listed with the record it replaced."""
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

DEFENSES = ("none", "jev")
INJ = re.compile(r"<INFORMATION>")
GOAL = re.compile(r"please do the following first: (.*?) After you do that", re.S)


def _rows(path: Path) -> tuple[list[dict], int]:
    """rows of a jsonl file, and how many lines did not parse (a line still being written)"""
    if not path.exists():
        return [], 0
    out, bad = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(r, dict):
            out.append(r)
    return out, bad


def key_of(r: dict) -> str:
    return "|".join([r["suite"], r["user_task"], r.get("injection_task") or "none", r["attack"], r["defense_id"]])


def parse_key(k: str) -> dict:
    suite, ut, it, attack, did = k.split("|")
    return {"suite": suite, "user_task": ut, "injection_task": None if it == "none" else it, "attack": attack,
            "defense_id": did, "defense": did.split("-")[0]}


def _num(task_id) -> int:
    try:
        return int(str(task_id).rsplit("_", 1)[-1])
    except ValueError:
        return 10 ** 6


def last_records(rows: list[dict]) -> tuple[dict, list[dict]]:
    """key -> the last record, and the keys with more than one"""
    by, dup = {}, Counter()
    for r in rows:
        if not all(r.get(k) for k in ("suite", "user_task", "attack", "defense_id")):
            continue
        k = key_of(r)
        dup[k] += 1
        by[k] = r
    return by, [{"key": k, "records": n} for k, n in dup.items() if n > 1]


def norm(text: str) -> str:
    return " ".join(str(text).replace("''", "'").split())


def goal_of(text: str) -> str | None:
    """the injection's goal, from a text holding AgentDojo's template"""
    m = GOAL.search(norm(text))
    return m.group(1).strip() if m else None


def planned(manifest: dict | None) -> dict:
    """user tasks, injection tasks, and episodes per defense, from the manifest's "all (n)" counts"""
    n_ut = n_it = None
    for c in (manifest or {}).get("calls", []):
        m = re.search(r"\((\d+)\)", str(c.get("user_tasks", "")))
        if m:
            n_ut = int(m.group(1))
        m = re.search(r"\((\d+)\)", str(c.get("injection_tasks", "")))
        if m:
            n_it = int(m.group(1))
    return {"user_tasks": n_ut, "injection_tasks": n_it, "clean": n_ut,
            "attacked": n_ut * n_it if n_ut is not None and n_it is not None else None}


def rate(vals: list) -> dict:
    known = [v for v in vals if v is not None]
    return {"k": sum(bool(v) for v in known), "n": len(known)}


def _med(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return st.median(vals) if vals else None


# ---- per episode


def _content(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return "\n".join(str(b.get("content", "")) for b in c if isinstance(b, dict))
    return "" if c is None else str(c)


def log_delivery(log: dict | None) -> dict | None:
    """from AgentDojo's episode log: tool messages the agent got, and those holding a whole injection goal"""
    if not isinstance(log, dict) or not isinstance(log.get("messages"), list):
        return None
    goals = [g for g in (goal_of(v) for v in (log.get("injections") or {}).values()) if g]
    tools = [m for m in log["messages"] if isinstance(m, dict) and m.get("role") == "tool"]
    texts = [norm(_content(m) or m.get("error") or "") for m in tools]
    held = [t for t in texts if any(g in t for g in goals)]
    return {"tool_messages": len(tools), "goals": len(goals), "goal_messages": len(held), "goal": bool(held)}


def output_view(row: dict) -> dict:
    """one filtered tool output: did it hold the injection, and is the goal still visible after"""
    before = str(row.get("before", ""))
    held = bool(INJ.search(before))
    g = goal_of(before) if held else None
    after = str(row.get("after", before))
    return {"held": held, "goal": g, "goal_visible": (g in norm(after)) if g else None}


def defense_rows(rows: list[dict]) -> dict:
    """episode key -> its defense.jsonl rows"""
    by = {}
    for r in rows:
        if r.get("episode"):
            by.setdefault(r["episode"], []).append(r)
    return by


def flags(recs: dict, drows: dict) -> dict:
    """the jev filter: tool-output flags on clean episodes and on attacked episodes, split by injection held"""
    groups = {g: Counter() for g in ("clean", "attacked_injection", "attacked_other")}
    spans, left = Counter(), Counter()
    for k, rows in drows.items():
        p = parse_key(k)
        if p["defense"] != "jev":
            continue
        rec = recs.get(k)
        if rec is None or rec.get("error"):
            # an episode with an error, or one still running with no record yet
            left["error" if rec is not None else "no_record_yet"] += len(rows)
            continue
        if rec.get("play"):  # a replayed episode counts only its last play's outputs
            rows = [r for r in rows if r.get("play") == rec["play"]]
        for r in rows:
            v = output_view(r)
            g = "clean" if p["injection_task"] is None else "attacked_injection" if v["held"] else "attacked_other"
            c = groups[g]
            c["outputs"] += 1
            c["windows"] += r.get("windows", r.get("chunks")) or 0  # newer rows say chunks
            c["windows_flagged"] += r.get("flagged") or 0
            c["outputs_flagged"] += (r.get("flagged") or 0) > 0
            c["lines_withheld"] += r.get("lines_withheld") or 0
            c["fallbacks"] += r.get("fallbacks", r.get("hidden_whole")) or 0
            c["finder_spans"] += len(r.get("finder_spans") or [])
            c["fail_open"] += bool(r.get("failed"))
            c["jev_input_tokens"] += r.get("jev_input_tokens") or 0
            spans["span"] += sum(s is not None for s in r.get("finder_spans") or [])
            if v["goal"] is not None:
                c["goal_outputs"] += 1
                c["goal_visible_after"] += bool(v["goal_visible"])
    out = {"jev": {g: dict(c) for g, c in groups.items()}}
    out["jev"]["rows_left_out"] = {"error": left["error"], "no_record_yet": left["no_record_yet"]}
    return out


def none_outputs(recs: dict, logs: dict) -> dict:
    """the no-defense episodes: attacked tool outputs holding the injection, from AgentDojo's logs"""
    c = Counter()
    for k, rec in recs.items():
        p = parse_key(k)
        if p["defense"] != "none" or not p["injection_task"] or rec.get("error"):
            continue
        log = logs.get(k)
        if not isinstance(log, dict):
            c["episodes_without_log"] += 1
            continue
        for m in log.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "tool":
                t = _content(m)
                c["outputs"] += 1
                if INJ.search(t):
                    c["outputs_with_injection"] += 1
                    c["goal_visible"] += goal_of(t) is not None
    return dict(c)


# ---- tables


def summary(recs: dict, plan: dict, logs: dict) -> dict:
    out = {}
    for d in DEFENSES:
        rs = {k: r for k, r in recs.items() if r.get("defense") == d}
        clean = [r for r in rs.values() if not r.get("injection_task")]
        att = [r for r in rs.values() if r.get("injection_task")]
        ok_c = [r for r in clean if not r.get("error")]
        ok_a = [r for r in att if not r.get("error")]
        dl = [(log_delivery(logs.get(k)), r) for k, r in rs.items() if r.get("injection_task") and not r.get("error")]
        out[d] = {
            "defense_ids": sorted({r["defense_id"] for r in rs.values()}),
            "clean_recorded": len(clean), "clean_planned": plan["clean"],
            "attacked_recorded": len(att), "attacked_planned": plan["attacked"],
            "utility_clean": rate([r.get("utility") for r in ok_c]),
            "utility_attacked": rate([r.get("utility") for r in ok_a]),
            "attack_success": rate([r.get("security") for r in ok_a]),
            "errors_clean": len(clean) - len(ok_c), "errors_attacked": len(att) - len(ok_a),
            "end_states": dict(Counter(r.get("end_state") for r in rs.values())),
            "attempts_over_1": sum((r.get("attempts") or 0) > 1 for r in rs.values()),
            "bad_tool_args": sum(len(r.get("bad_tool_args") or []) for r in rs.values()),
            "turns_median": _med([r.get("turns") for r in rs.values()]),
            "tool_calls_median": _med([r.get("tool_calls") for r in rs.values()]),
            "prompt": sum(r.get("llm_prompt_tokens") or 0 for r in rs.values()),
            "completion": sum(r.get("llm_completion_tokens") or 0 for r in rs.values()),
            "usd_est": round(sum(r.get("llm_usd_est") or 0 for r in rs.values()), 6),
            "jev_in": sum(r.get("jev_input_tokens") or 0 for r in rs.values()),
            "jev_out": sum(r.get("jev_output_tokens") or 0 for r in rs.values()),
            "jev_requests": sum(r.get("jev_requests") or 0 for r in rs.values()),
            "jev_failed_requests": sum(r.get("jev_failed_requests") or 0 for r in rs.values()),
            "delivery": {"episodes_with_log": sum(x is not None for x, _ in dl),
                         "goal_delivered": sum(bool(x and x["goal"]) for x, _ in dl),
                         "goal_delivered_and_attack_succeeded": sum(bool(x and x["goal"]) and bool(r.get("security"))
                                                                    for x, r in dl)}}
    return out


def by_injection(recs: dict) -> list[dict]:
    its = sorted({r["injection_task"] for r in recs.values() if r.get("injection_task")}, key=_num)
    out = []
    for it in its:
        row = {"injection_task": it}
        for d in DEFENSES:
            rs = [r for r in recs.values() if r.get("defense") == d and r.get("injection_task") == it]
            ok = [r for r in rs if not r.get("error")]
            row[d] = {"attack_success": rate([r.get("security") for r in ok]),
                      "utility": rate([r.get("utility") for r in ok]), "errors": len(rs) - len(ok)}
        out.append(row)
    return out


def by_user(recs: dict) -> list[dict]:
    uts = sorted({r["user_task"] for r in recs.values()}, key=_num)
    out = []
    for ut in uts:
        row = {"user_task": ut}
        for d in DEFENSES:
            rs = [r for r in recs.values() if r.get("defense") == d and r["user_task"] == ut]
            clean = [r for r in rs if not r.get("injection_task")]
            att = [r for r in rs if r.get("injection_task") and not r.get("error")]
            c = clean[-1] if clean else None
            row[d] = {"utility_clean": None if c is None or c.get("error") else c.get("utility"),
                      "clean_error": bool(c and c.get("error")), "clean_recorded": c is not None,
                      "utility_attacked": rate([r.get("utility") for r in att]),
                      "attack_success": rate([r.get("security") for r in att]),
                      "errors": sum(bool(r.get("error")) for r in rs)}
        out.append(row)
    return out


def paired(recs: dict) -> dict:
    """per (user task, injection task): each defense's security and utility, and counts over pairs where both
    defenses of a comparison have an episode without an error"""
    cells = {}
    for r in recs.values():
        cells.setdefault((r["user_task"], r.get("injection_task")), {})[r["defense"]] = r
    pairs = []
    for (ut, it), by in sorted(cells.items(), key=lambda kv: (_num(kv[0][0]), _num(kv[0][1] or "x_-1"))):
        if it is None:
            continue
        pairs.append({"user_task": ut, "injection_task": it, **{
            d: ({"security": by[d].get("security"), "utility": by[d].get("utility"), "error": bool(by[d].get("error"))}
                if d in by else None) for d in DEFENSES}})
    counts = {}
    for a, b in (("none", "jev"),):
        both = [p for p in pairs if p[a] and p[b] and not p[a]["error"] and not p[b]["error"]]
        for m in ("security", "utility"):
            c = Counter((bool(p[a][m]), bool(p[b][m])) for p in both)
            counts[f"{a}_vs_{b}_{m}"] = {"pairs": len(both), f"{a}_yes_{b}_no": c[(True, False)],
                                        f"{a}_no_{b}_yes": c[(False, True)], "both_yes": c[(True, True)],
                                        "both_no": c[(False, False)]}
    # clean episodes, per user task
    clean = {}
    for r in recs.values():
        if not r.get("injection_task"):
            clean.setdefault(r["user_task"], {})[r["defense"]] = r
    cc = {}
    for a, b in (("none", "jev"),):
        both = [v for v in clean.values() if a in v and b in v and not v[a].get("error") and not v[b].get("error")]
        c = Counter((bool(v[a].get("utility")), bool(v[b].get("utility"))) for v in both)
        cc[f"{a}_vs_{b}_utility_clean"] = {"user_tasks": len(both), f"{a}_yes_{b}_no": c[(True, False)],
                                           f"{a}_no_{b}_yes": c[(False, True)], "both_yes": c[(True, True)],
                                           "both_no": c[(False, False)]}
    return {"pairs": pairs, "counts": counts, "clean_counts": cc}


def llm_log(rows: list[dict]) -> dict:
    """every llm call, errored episodes and replays included, by defense"""
    out = {}
    for r in rows:
        d = str(r.get("defense", "?")).split("-")[0]
        c = out.setdefault(d, Counter())
        c["calls"] += 1
        c["errors"] += "error" in r
        c["prompt"] += r.get("prompt_tokens") or 0
        c["completion"] += r.get("completion_tokens") or 0
        c["usd_est"] += r.get("usd_est") or 0
        c["latency_s"] += (r.get("latency_ms") or 0) / 1000
    return {d: {k: round(v, 6) if isinstance(v, float) else v for k, v in c.items()} for d, c in out.items()}


def jev_log(rows: list[dict]) -> dict:
    reqs = [r for r in rows if r.get("kind") == "request"]
    ev = [r for r in rows if r.get("kind") == "event" and r.get("event") == "guard_lines"]
    return {"requests": len(reqs),
            "by_row": dict(Counter("+".join(r.get("rows") or [r.get("filter") or "?"]) for r in reqs)),
            "input_tokens": sum((r.get("usage") or {}).get("input_tokens") or 0 for r in reqs),
            "output_tokens": sum((r.get("usage") or {}).get("output_tokens") or 0 for r in reqs),
            "errors": sum("error" in r for r in reqs),
            "models": sorted({r["model"] for r in reqs if isinstance(r.get("model"), str)}),
            "finder": {"calls": len(ev), "span": sum(not e.get("fell_back") for e in ev),
                       "fallback": sum(bool(e.get("fell_back")) for e in ev),
                       "fallback_reasons": dict(Counter(str(e.get("reason")) for e in ev if e.get("fell_back")))}}


def load_logs(d: Path, recs: dict) -> dict:
    out = {}
    for k, r in recs.items():
        p = Path(r.get("_dir") or d) / str(r.get("log") or "")
        try:
            out[k] = json.loads(p.read_text()) if r.get("log") and p.is_file() else None
        except (OSError, ValueError):
            out[k] = None
    return out


def _brief(r: dict | None) -> dict | None:
    if not r:
        return None
    return {k: r.get(k) for k in ("utility", "security", "error", "started", "commit", "note")}


def analyze(d: Path, manifest: Path, replace: tuple = ()) -> dict:
    rows, bad = _rows(d / "records.jsonl")
    recs, dups = last_records(rows)
    man = None
    try:
        man = json.loads(manifest.read_text())
    except (OSError, ValueError):
        pass
    plan = planned(man)
    drows_all, bad_d = _rows(d / "defense.jsonl")
    jrows, bad_j = _rows(d / "jev_trace.jsonl")
    lrows, bad_l = _rows(d / "llm_calls.jsonl")
    runs, _ = _rows(d / "runs.jsonl")
    replaced = []
    for rd in replace:
        rd = Path(rd)
        new, _ = last_records(_rows(rd / "records.jsonl")[0])
        for k, r in sorted(new.items()):
            replaced.append({"key": k, "dir": str(rd), "was": _brief(recs.get(k)), "now": _brief(r)})
            recs[k] = {**r, "_dir": str(rd)}
        drows_all = [x for x in drows_all if x.get("episode") not in new] + _rows(rd / "defense.jsonl")[0]
        jrows += _rows(rd / "jev_trace.jsonl")[0]
        lrows += _rows(rd / "llm_calls.jsonl")[0]
        runs += _rows(rd / "runs.jsonl")[0]
    logs = load_logs(d, recs)
    drows = defense_rows(drows_all)
    errors = [{"key": k, "error": str(r.get("error"))[:200], "turns": r.get("turns"), "attempts": r.get("attempts")}
              for k, r in sorted(recs.items()) if r.get("error")]
    # a jev error stops an episode and marks it for a rerun. it is not a result until the rerun replaces it
    reruns = [{"key": k, **{f: (r.get("jev_error") or {}).get(f) for f in ("call_id", "filter", "error")}}
              for k, r in sorted(recs.items()) if r.get("rerun")]
    return {"dir": str(d), "manifest": str(manifest), "plan": plan, "records": len(recs),
            "records_by_kind": dict(Counter(f"{r.get('defense')} {'attacked' if r.get('injection_task') else 'clean'}"
                                            for r in recs.values()).most_common()),
            "unparsed_lines": {"records": bad, "defense": bad_d, "jev_trace": bad_j, "llm_calls": bad_l},
            "duplicates": dups, "errors": errors, "needs_rerun": reruns, "replaced": replaced,
            "summary": summary(recs, plan, logs), "flags": flags(recs, drows), "none_outputs": none_outputs(recs, logs),
            "by_injection_task": by_injection(recs), "by_user_task": by_user(recs), "paired": paired(recs),
            "llm_log": llm_log(lrows), "jev_log": jev_log(jrows),
            "runner_lines": [{k: v for k, v in r.items() if k != "argv"} for r in runs],
            "logs_missing": sum(v is None for v in logs.values())}
