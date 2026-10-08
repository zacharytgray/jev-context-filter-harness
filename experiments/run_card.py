"""run card: one run's tool calls, what each filter did to each output, and where its prompt tokens went. also a
roll-up of a runs folder by configuration and task.

    PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 python experiments/run_card.py <run dir or rep dir>
    PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 python experiments/run_card.py --rollup experiments/runs-<tag>/<test> \
        [--price-in 0.14] [--price-out 0.40] [--json out.json]

offline and read only. a card reads a run's levers.json, result.json, transcript.json, llm.jsonl and trace.jsonl, and
its graded row in the nearest records.jsonl. the roll-up reads only records.jsonl files. calls no model. writes
nothing unless --json is given.

calls. every trace row of a tool call carries the turn and the call id, so a call's output or result event, its chunk
rows, its jev requests, its jev fallbacks and a jev error are matched by (turn, call id). a request row's attempts
give its retries; a jev_fallback event is a request that failed every attempt, so its filter acted as if off. sizes come from the output event: raw (the whole
output), after relevance, after injection, after the cap, and what entered (the transcript's own length). kept and
flagged counts and p come from the chunk rows. hidden ranges and reasons come from the output event.

tokens. llm.jsonl has prompt_tokens and completion_tokens per llm call, never per message. call 1's prompt is the
fixed prefix: system prompt, tool schemas and request. the transcript only grows, so the rise from one call to the next
is what that turn added: the assistant message, counted at its completion_tokens, and the turn's tool outputs, the
rest (split by chars when a turn made several calls). a message's contribution is its tokens times the later calls that
re-sent it. prefix times calls plus every contribution equals the summed prompt_tokens. outputs of the last turn are
never re-sent, so their size is a chars/4 estimate and their contribution is 0.

compaction. a compaction replaces the request and everything after it, up to the newest exchange, with one user
message, the request then the summary. the newest exchange (the agent's last message and its tool results) stays.
transcript.json keeps every message that entered context, each summary message just before the exchange kept with it;
the compaction event's at is that message's index, and a turn call after it saw the system prompt, that message, then
transcript[at + 1:]. the calls between two compactions form a segment and the rise rule runs inside it. a segment's
first prompt less call 1's is the summary plus the kept exchange, split between them by chars and counted times the
segment's calls. outputs the summary replaced were never re-sent whole: chars/4, contribution 0. the compaction
calls' own tokens are counted apart, and the cost is on agent plus compaction.

re-fetch. a read's output event lists the file lines it showed and the ones it hid, with why. a later read of the same
path that shows lines an earlier read hid is a re-fetch of hidden lines; one that shows lines shown before is a plain
re-fetch. bash output rows are not file lines, so a bash call counts only when it repeats an earlier command exactly.
an edit or write of the path in between shifts line numbers, so those are marked "after edit".
"""
import argparse
import json
import re
import statistics as st
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from metrics import run_metrics  # noqa: E402
from s1harness.view import _record  # noqa: E402

CHARS_PER_TOKEN = 4  # only where no recorded count exists, and always marked
PRICE_IN, PRICE_OUT = 0.14, 0.40  # the roll-up's llm dollars per 1M prompt and completion tokens
CONFIGS = ("plain", "relevance", "injection", "relevance-injection")
STAGES = ("raw_chars", "after_relevance", "after_injection", "after_cap", "entered_chars")
SHORT = {"relevance": "rel", "cap": "cap", "injection": "inj"}
# the roll-up's detail columns: numeric keys check.py wrote ending in one of these, so counts such as lb_quotes or
# cb_decl_file_pred stay out. score has its own column. never run.py's own keys or a key metrics.py writes (a missing
# run dir still gives all of them)
SCORE = re.compile(r"(^|_)(score|recall|precision|f1|accuracy)$")
RUNNER_KEYS = {"rep", "wall_s", "exit"}
METRIC_KEYS = set(run_metrics(Path("/nonexistent/run")))


def jl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:  # a half-written last line
            pass
    return out


def jf(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def norm(p) -> str:
    p = str(p or "").strip().strip("'\"")
    while p.startswith("./"):
        p = p[2:]
    return p.rstrip("/") or "."


def fmt(n) -> str:
    return "-" if n is None else f"{n:,.0f}"


def kfmt(n) -> str:
    if n is None:
        return "-"
    return f"{n / 1000:,.0f}k" if abs(n) >= 10000 else f"{n / 1000:,.1f}k" if abs(n) >= 1000 else f"{n:,.0f}"


def pct(a, b) -> str:
    return "-" if not b else f"{100 * a / b:.0f}%"


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def short(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "~"


def yn(v) -> str:
    return "?" if v is None else "yes" if v else "no"


def _set(ranges) -> set:
    return {n for a, b in ranges for n in range(a, b + 1)}


def span(a: int, b: int) -> str:
    return str(a) if a == b else f"{a}-{b}"


# ---------------------------------------------------------------- loading


def run_dir_of(d) -> Path:
    """a rep dir (run.py's <arm>/<task>/r<k>) holds its run under run/<timestamp>"""
    d = Path(d)
    if not (d / "trace.jsonl").exists() and (d / "run").is_dir():
        subs = sorted(p for p in (d / "run").iterdir() if p.is_dir())
        if subs:
            return subs[-1]
    return d


def task_spec(rec: dict) -> dict:
    if not rec.get("test") or not rec.get("task"):
        return {}
    p = EXP / str(rec["test"]) / "tasks" / str(rec["task"]) / "task.toml"
    return tomllib.loads(p.read_text()) if p.exists() else {}


def load(run_dir, rec: dict | None = None) -> dict:
    d = run_dir_of(run_dir)
    rec = rec if rec is not None else (_record(d) or {})
    trace = jl(d / "trace.jsonl")
    run = dict(dir=d, rec=rec, spec=task_spec(rec), levers=jf(d / "levers.json") or {},
               result=jf(d / "result.json") or {}, msgs=jf(d / "transcript.json") or [],
               llm=[r for r in jl(d / "llm.jsonl") if r.get("kind") == "turn"], trace=trace,
               comp_llm=[r for r in jl(d / "llm.jsonl") if r.get("kind") == "compaction"],
               llm_retries=[r for r in jl(d / "llm.jsonl") if r.get("kind") == "retry"],
               compactions=[r for r in trace if r.get("event") == "compaction"],
               jev_errors=[r for r in trace if r.get("event") == "jev_error"],
               jev_fallbacks=[r for r in trace if r.get("event") == "jev_fallback"], warnings=[])
    run["calls"] = calls(run)
    tokens(run)
    refetches(run)
    return run


def calls(run: dict) -> list[dict]:
    """one row per tool call, in transcript order, with its trace rows matched by turn and call id"""
    rows = defaultdict(lambda: dict(out=None, res=None, chunks=[], reqs=[], jev_error=None, fallbacks=[]))
    for r in run["trace"]:
        if r.get("call_id") is None:
            continue
        b = rows[(r.get("turn"), r["call_id"])]
        if r.get("kind") == "request":
            b["reqs"].append(r)
        elif r.get("kind") == "chunk":
            b["chunks"].append(r)
        elif r.get("event") == "jev_fallback":
            b["fallbacks"].append(r)
        elif r.get("event") in ("output", "result", "jev_error"):
            b[{"output": "out", "result": "res", "jev_error": "jev_error"}[r["event"]]] = r
    cs, turn, pending = [], 0, []
    for j, m in enumerate(run["msgs"]):
        if m["role"] == "assistant":
            turn += 1
            pending = []
            for tc in m.get("tool_calls") or []:
                raw = tc["function"].get("arguments") or ""
                try:
                    args = json.loads(raw or "{}")
                except json.JSONDecodeError:
                    args = {}
                c = dict(turn=turn, id=tc.get("id", ""), tool=tc["function"]["name"],
                         args=args if isinstance(args, dict) else {}, text=None, msg=None)
                cs.append(c)
                pending.append(c)
        elif m["role"] == "tool":  # results follow their turn's calls, in order
            c = next((c for c in pending if c["id"] == m.get("tool_call_id") and c["msg"] is None), None)
            if c is not None:
                c.update(text=m.get("content") or "", msg=j)
    matched = set()
    for c in cs:
        key = (c["turn"], c["id"])
        c.update(rows.get(key) or dict(out=None, res=None, chunks=[], reqs=[], jev_error=None, fallbacks=[]))
        matched.add(key)
        describe(c, run)
    for key in rows:
        if key not in matched:
            run["warnings"].append(f"trace rows for turn {key[0]} call {key[1]} match no call in the transcript")
    if len(run["llm"]) != turn:
        run["warnings"].append(f"{len(run['llm'])} llm calls, {turn} assistant messages")
    return cs


def describe(c: dict, run: dict):
    o, r = c["out"], c["res"]
    ev = o or r
    c["seen"] = None if c["text"] is None else len(c["text"])
    if ev is not None and c["seen"] is not None and ev["entered_chars"] != c["seen"]:
        run["warnings"].append(f"t{c['turn']} {c['id']}: {c['seen']} chars in the transcript, "
                               f"{ev['entered_chars']} entered")
    c["raw"] = ev["raw_chars"] if ev else None
    c["stages"] = [o[k] for k in STAGES] if o else None
    c["rel_rows"] = [x for x in c["chunks"] if x["filter"] == "relevance"]
    c["inj_rows"] = [x for x in c["chunks"] if x["filter"] == "injection"]
    c["kept"] = sum(x["decision"] == "kept" for x in c["rel_rows"])
    c["flagged"] = sum(x["decision"] == "flagged" for x in c["inj_rows"])
    c["jev_n"] = len(c["reqs"])
    c["jev_in"] = sum((q.get("usage") or {}).get("input_tokens") or 0 for q in c["reqs"])
    c["jev_failed"] = sum(bool(q.get("error")) for q in c["reqs"])  # failed every attempt
    c["jev_retries"] = sum((q.get("attempts") or 1) - 1 for q in c["reqs"])
    c["jev_ms"] = o["jev_ms"] if o else None
    c["tool_ms"] = o["tool_ms"] if o else None
    if o and (o["jev_requests"], o["jev_input_tokens"]) != (c["jev_n"], c["jev_in"]):
        run["warnings"].append(f"t{c['turn']} {c['id']}: the output event counts {o['jev_requests']} jev requests and "
                               f"{o['jev_input_tokens']} tokens, the request rows {c['jev_n']} and {c['jev_in']}")
    c["hidden"] = o["hidden"] if o else []
    why = []
    if c["rel_rows"]:
        why.append(f"rel kept {c['kept']}/{len(c['rel_rows'])}")
    elif o and o["ok"] and run["levers"].get("relevance") and o["pass_through"]:
        why.append("short, whole")
    if c["inj_rows"]:
        why.append(f"inj flagged {c['flagged']}/{len(c['inj_rows'])}")
        stuck = sum(x.get("outcome") == "stuck" for x in c["inj_rows"])
        if stuck:
            why.append(f"inj stuck {stuck}")
    if o and o.get("skipped_past_cap"):  # chunks the injection filter never asked, past the cap
        why.append(f"inj not asked {o['skipped_past_cap']} past the cap")
    by = defaultdict(list)
    for h in c["hidden"]:
        by[h["reason"]].append(h)
    for reason, hs in by.items():
        n = sum(h["last"] - h["first"] + 1 for h in hs)
        rs = ", ".join(span(h["first"], h["last"]) for h in hs[:3]) + (f", +{len(hs) - 3}" if len(hs) > 3 else "")
        why.append(f"hid {SHORT.get(reason, reason)} {n} line{'s' if n != 1 else ''} ({rs})")
    if o and o["cap"]:
        why.append(f"cap at line {o['cap']['last_line']}/{o['cap']['total']}")
    if r and r.get("error"):
        why.append("not run: " + short(c["text"] or "", 40))
    elif o and not o["ok"]:
        why.append("tool error: " + short(c["text"] or "", 40))
    elif ev is None:
        why.append("nothing entered: the run stopped")
    c["why"] = "; ".join(why) or "-"
    issues = []
    if c["jev_failed"]:
        issues.append(f"{c['jev_failed']} failed request{'s' if c['jev_failed'] != 1 else ''}")
    if c["jev_retries"]:
        issues.append(f"{c['jev_retries']} retr{'ies' if c['jev_retries'] != 1 else 'y'}")
    if c["fallbacks"]:
        by = Counter(f["filter"] for f in c["fallbacks"])
        issues.append("fell back: " + ", ".join(f"{k} {v}" for k, v in by.items()))
    if c["jev_error"]:
        issues.append(f"jev error in {c['jev_error'].get('filter')}: run stopped")
    c["issues"] = ", ".join(issues) or "-"


# ---------------------------------------------------------------- tokens


def tokens(run: dict):
    """per-message tokens from the rise between llm calls, and each message's contribution (tokens x re-sends).
    a compaction resets the transcript: the rise rule runs inside each segment of calls between compactions"""
    llm, msgs, cs = run["llm"], run["msgs"], run["calls"]
    n = len(llm)
    P = [r.get("prompt_tokens") or 0 for r in llm]
    C = [r.get("completion_tokens") or 0 for r in llm]
    # each call's compaction (the last one at or before its turn), then its messages as transcript indices
    comps = sorted(run.get("compactions") or [], key=lambda e: e["at"])
    seg = [max((k for k, e in enumerate(comps) if e["turn"] <= (r.get("turn") or 0)), default=-1) for r in llm]
    ns = [r["n_messages"] + (comps[g]["at"] - 1 if g >= 0 else 0) for r, g in zip(llm, seg)]
    run.update(n=n, total=sum(P), prefix=P[0] if P else 0)
    at = {c["msg"]: c for c in cs if c["msg"] is not None}
    agent = other = summ = 0.0
    for c in cs:
        c.update(tok=None, tok_est=True, resent=0, contrib=0.0)
    ends = {}  # a segment's first call and its end, exclusive
    for i, g in enumerate(seg):
        ends.setdefault(g, [i, i])[1] = i + 1

    def mchars(m: dict) -> int:
        return len(m.get("content") or "") + sum(len(tc["function"].get("arguments") or "") +
                                                 len(tc["function"]["name"]) for tc in m.get("tool_calls") or [])

    for g, (first, end) in ends.items():
        if g < 0:
            continue
        # this segment's first prompt less the fixed prefix is the summary and the exchange kept with it, re-sent by
        # every call in the segment. split by chars; the request in the summary message is already in the prefix
        a = comps[g]["at"]
        kept = range(a + 1, min(ns[first], len(msgs)))
        sizes = {j: mchars(msgs[j]) for j in kept}
        s_chars = len(msgs[a].get("content") or "") - len(msgs[1].get("content") or "")
        whole = s_chars + sum(sizes.values())
        rise = P[first] - run["prefix"]
        r = end - first
        summ += (rise * s_chars / whole if whole else rise) * r
        for j, size in sizes.items():
            share = rise * size / whole
            if msgs[j]["role"] == "tool" and at.get(j) is not None:
                at[j].update(tok=share, tok_est=True, resent=r, contrib=share * r)
            elif msgs[j]["role"] == "assistant":
                agent += share * r
            else:
                other += share * r
    for i in range(1, n):
        if seg[i] != seg[i - 1]:
            continue
        end = ends[seg[i]][1]
        span = range(ns[i - 1], min(ns[i], len(msgs)))
        delta = P[i] - P[i - 1]
        a_tok = min(C[i - 1], delta)
        rest = delta - a_tok
        tools = [j for j in span if msgs[j]["role"] == "tool"]
        others = [j for j in span if msgs[j]["role"] not in ("tool", "assistant")]
        tchars = sum(len(msgs[j].get("content") or "") for j in tools)
        r = end - i  # calls i+1..end (1-based) of this segment re-send what call i's turn added
        agent += a_tok * r
        if not tools and not others:  # nothing else was added, so the rest is the assistant's too
            agent += rest * r
        for j in tools:
            # outputs that are all empty share the rest evenly, so the sum stays exact
            tok = rest * (len(msgs[j].get("content") or "") / tchars if tchars else 1 / len(tools))
            c = at.get(j)
            if c is not None:
                c.update(tok=tok, tok_est=len(tools) > 1, resent=r, contrib=tok * r)
        if others and not tools:
            other += rest * r
    for c in cs:  # never re-sent whole: the last turn's outputs, and the outputs a summary replaced
        if c["tok"] is None:
            c["tok"] = len(c["text"] or "") / CHARS_PER_TOKEN
    run["agent_contrib"] = agent
    run["tool_contrib"] = sum(c["contrib"] for c in cs)
    run["other_contrib"] = other
    run["summary_contrib"] = summ
    run["prefix_contrib"] = run["prefix"] * n
    run["attributed"] = run["prefix_contrib"] + agent + run["tool_contrib"] + other + summ
    # an independent estimate: every re-sent message at chars/4, the prefix as recorded
    est = run["prefix_contrib"]
    for g, (first, end) in ends.items():
        if g >= 0:
            # the request in that message is already in the prefix
            a = comps[g]["at"]
            summ_chars = len(msgs[a].get("content") or "") - len(msgs[1].get("content") or "")
            summ_chars += sum(mchars(m) for m in msgs[a + 1:ns[first]])
            est += summ_chars / CHARS_PER_TOKEN * (end - first)
    for i in range(1, n):
        if seg[i] != seg[i - 1]:
            continue
        for m in msgs[ns[i - 1]:ns[i]]:
            est += mchars(m) / CHARS_PER_TOKEN * (ends[seg[i]][1] - i)
    run["est_total"] = est
    run["est_tool"] = sum(len(c["text"] or "") / CHARS_PER_TOKEN * c["resent"] for c in cs)
    run["comp_prompt"] = sum(r.get("prompt_tokens") or 0 for r in run.get("comp_llm") or [])
    run["comp_completion"] = sum(r.get("completion_tokens") or 0 for r in run.get("comp_llm") or [])


# ---------------------------------------------------------------- re-fetches


def refetches(run: dict):
    """a later read of lines an earlier read of the same path showed or hid, or bash repeating a command"""
    cs = run["calls"]
    edits = defaultdict(list)
    for k, c in enumerate(cs):
        if c["tool"] in ("edit", "write") and c["text"] is not None and not c["text"].startswith("Error"):
            edits[norm(c["args"].get("path"))].append(k)
    fetched = []
    stats = Counter()
    for c in cs:
        c.update(refetch=[], re=None)
    for k, c in enumerate(cs):
        o = c["out"]
        if not o or not o["ok"]:
            continue
        key = ("read", norm(o.get("path"))) if c["tool"] == "read" else ("bash", o.get("command"))
        shown = _set(o["shown"])
        hidden = {n: h["reason"] for h in o["hidden"] for n in range(h["first"], h["last"] + 1)}
        before = [f for f in fetched if f["key"] == key]
        if before:
            was_shown = set().union(*(f["shown"] for f in before))
            was_hidden = {n: why for f in before for n, why in f["hidden"].items() if n not in was_shown}
            got = shown & set(was_hidden)
            if key[0] == "bash" or shown & (was_shown | set(was_hidden)):
                reasons = sorted({was_hidden[n] for n in got})
                after_edit = key[0] == "read" and any(before[-1]["k"] < e < k for e in edits[key[1]])
                c["re"] = "hidden" if got else "seen"
                stats["all"] += 1
                stats["hidden"] += bool(got)
                stats["after_edit"] += after_edit
                for x in reasons:
                    stats[x] += 1
                tag = (f"t{c['turn']} {c['tool']}" + ("*" + "/".join(SHORT.get(x, x) for x in reasons) if got else "")
                       + (" after edit" if after_edit else ""))
                for f in before:
                    if key[0] == "bash" or shown & (f["shown"] | set(f["hidden"])):
                        cs[f["k"]]["refetch"].append(tag)
        fetched.append(dict(k=k, key=key, shown=shown, hidden=hidden))
    run["refetch"] = stats


# ---------------------------------------------------------------- the card


def args_short(c: dict) -> str:
    a = c["args"]
    if c["tool"] == "read":
        s = norm(a.get("path"))
        if a.get("start_line") or a.get("end_line"):
            s += f":{a.get('start_line') or ''}-{a.get('end_line') or ''}"
        return s + (f' q="{short(a["question"], 30)}"' if a.get("question") else "")
    if c["tool"] == "bash":
        s = short(str(a.get("command") or "").split("\n")[0], 40)
        return s + (f' q="{short(a["question"], 24)}"' if a.get("question") else "")
    if c["tool"] in ("list", "edit", "write"):
        return norm(a.get("path"))
    if c["tool"] == "ask_user":
        return f'"{short(a.get("question"), 44)}"'
    return short(json.dumps(a), 44)


def summary(run: dict) -> dict:
    rec, result = run["rec"], run["result"]
    kind = (run["spec"].get("labels") or {}).get("kind")
    attack = kind == "attack" if kind else rec.get("harm") is not None
    cs = run["calls"]
    top = sorted(cs, key=lambda c: -c["contrib"])[:3]
    reqs = [r for r in run["trace"] if r.get("kind") == "request"]
    reads = [c for c in cs if c["tool"] == "read" and c["out"] and c["out"]["ok"]]
    return dict(
        configuration=run["levers"].get("name") or rec.get("configuration") or rec.get("arm"), task=rec.get("task"),
        rep=rec.get("rep"), turns=result.get("turns", rec.get("turns")), calls=run["n"],
        end=result.get("end", rec.get("end")), rerun=bool(result.get("rerun") or rec.get("rerun")),
        score=rec.get("score") if is_num(rec.get("score")) else None, harm=rec.get("harm") if attack else None, attack=attack, total=run["total"],
        recorded=rec.get("llm_prompt_tokens"), jev_in=sum((r.get("usage") or {}).get("input_tokens") or 0 for r in reqs),
        jev_in_rec=rec.get("jev_input_tokens"), jev_n=len(reqs), jev_failed=sum(bool(r.get("error")) for r in reqs),
        jev_errors=[{k: e.get(k) for k in ("turn", "call_id", "filter", "error")} for e in run["jev_errors"]],
        jev_retries=sum((r.get("attempts") or 1) - 1 for r in reqs),
        skipped_past_cap=sum((c["out"] or {}).get("skipped_past_cap") or 0 for c in cs),
        llm_retries=len(run.get("llm_retries") or []),
        llm_retry_wait_s=sum(r.get("wait_s") or 0 for r in run.get("llm_retries") or []),
        jev_fallbacks={f: sum(e.get("filter") == f for e in run["jev_fallbacks"]) for f in ("relevance", "injection", "finder")},
        prefix=run["prefix_contrib"], agent=run["agent_contrib"], tool=run["tool_contrib"], other=run["other_contrib"],
        top3=sum(c["contrib"] for c in top), top=[(c["turn"], c["tool"], args_short(c), c["contrib"]) for c in top],
        reads=len(reads), raw_per_read=med([c["raw"] for c in reads]), seen_per_read=med([c["seen"] for c in reads]),
        refetch=run["refetch"]["all"], refetch_hidden=run["refetch"]["hidden"],
        refetch_edit=run["refetch"]["after_edit"],
        refetch_by=({SHORT[k]: run["refetch"][k] for k in SHORT if run["refetch"][k]}),
        compactions=len(run["compactions"]), summary=run["summary_contrib"], comp_prompt=run["comp_prompt"],
        comp_completion=run["comp_completion"], completion=sum(r.get("completion_tokens") or 0 for r in run["llm"]),
        warnings=run["warnings"])


def pf(x: dict) -> str:
    """a chunk row's p, "fallback" marked when a request on it fell back (p is none when no part was answered)"""
    p = "p-" if x["p"] is None else f"p{x['p']:.2f}"
    return p + (" (fallback)" if x.get("fallback") else "")


def chunk_lines(c: dict) -> list[str]:
    """what the filters decided on each chunk of one call. p is jev's; the agent never sees it"""
    out = []
    if c["rel_rows"]:
        kept = [x for x in c["rel_rows"] if x["decision"] == "kept"]
        dropped = [x for x in c["rel_rows"] if x["decision"] != "kept"]
        s = ", ".join(f"{x['lines']} {pf(x)}" for x in kept[:6]) + (f", +{len(kept) - 6}" if len(kept) > 6 else "")
        d = f". dropped {len(dropped)}, highest p {max(x['p'] for x in dropped):.2f}" if dropped else ""
        out.append(f"relevance, kept {len(kept)} of {len(c['rel_rows'])}: {s or 'none'}{d}")
    if c["inj_rows"]:
        flagged = [x for x in c["inj_rows"] if x["decision"] == "flagged"]
        clear = [x for x in c["inj_rows"] if x["decision"] != "flagged"]
        parts = []
        for x in flagged:
            h = x.get("hidden") or []
            h = [h] if h and isinstance(h[0], int) else h  # runs before the loop kept one range
            if "outcome" in x:  # the loop: the final outcome says whether every line went, not the first round's span
                how = f"whole, {x['outcome']}" if x["outcome"] in ("no span", "stuck", "long line") else "span"
            else:
                how = "span" if x.get("span") else f"whole, {x.get('reason') or 'no span'}"
            loop = (f", {x['outcome']} after {len(x['rounds'])} round{'s' if len(x['rounds']) != 1 else ''}, "
                    f"{x['calls']} calls") if "outcome" in x else ""
            parts.append(f"{x['lines']} {pf(x)}, hid {', '.join(span(*r) for r in h)} ({how}){loop}" if h
                         else f"{x['lines']} {pf(x)}{loop}")
        ps = [x["p"] for x in clear if x["p"] is not None]
        fb = sum(bool(x.get("fallback")) for x in clear)
        d = (f". clear {len(clear)}" + (f", highest p {max(ps):.2f}" if ps else "")
             + (f", {fb} by a fallback" if fb else "")) if clear else ""
        out.append(f"injection, flagged {len(flagged)} of {len(c['inj_rows'])}: {'; '.join(parts) or 'none'}{d}")
    return out


def card(run: dict) -> str:
    s = summary(run)
    lev = run["levers"]
    T = run["total"]
    L = [f"run card: {s['configuration']} / {s['task'] or '?'} / r{'?' if s['rep'] is None else s['rep']}"
         f"   ({run['dir']})",
         "filters: " + ", ".join(f"{k} {'on' if lev.get(k) else 'off'}" for k in ("relevance", "injection"))
         + ", cap always on",
         "", "1. tool calls"]
    hdr = (f"{'turn':>4} {'tool':5} {'args':46} {'raw':>8} {'rel':>8} {'inj':>8} {'cap':>8} {'seen':>8} {'tok':>7} "
           f"{'x':>3} {'contrib':>9} {'jev req':>7} {'jev in':>7} {'jev ms':>6} {'tool ms':>7}  "
           "filters  |  re-fetch  |  jev issues")
    L += [hdr, "-" * len(hdr)]
    for c in run["calls"]:
        rel, inj, cap = c["stages"][1:4] if c["stages"] else (None, None, None)
        tok = f"{c['tok']:,.0f}{'~' if c['tok_est'] else ''}"
        L.append(f"{c['turn']:>4} {c['tool'][:5]:5} {args_short(c)[:46]:46} {fmt(c['raw']):>8} {fmt(rel):>8} "
                 f"{fmt(inj):>8} {fmt(cap):>8} {fmt(c['seen']):>8} {tok:>7} {c['resent']:>3} {c['contrib']:>9,.0f} "
                 f"{c['jev_n'] or '-':>7} {fmt(c['jev_in']) if c['jev_n'] else '-':>7} {fmt(c['jev_ms'] or None):>6} "
                 f"{fmt(c['tool_ms']):>7}  {c['why']}  |  {', '.join(c['refetch']) or '-'}  |  {c['issues']}")
    L += ["",
          "   raw: the whole output, before any filter. rel, inj, cap: chars after each stage (the order the stages",
          "   run in). seen: what entered context, the output whole after the filters. tok: this output's tokens from",
          "   the rise in prompt_tokens (see 3); '~' is an estimate: chars/4 for an output never re-sent, or a char split",
          "   when one turn made several calls. x: later calls that re-sent it. contrib = tok x x. jev ms: time in the",
          "   filters' jev calls. re-fetch: later calls that fetched these lines again; * marks lines this call hid,",
          "   with the filter that hid them."]
    if run["warnings"]:
        L.append("   warnings: " + "; ".join(run["warnings"]))
    detail = [(c, chunk_lines(c)) for c in run["calls"]]
    detail = [(c, ls) for c, ls in detail if ls]
    if detail:
        L += ["", "2. what the filters decided, per chunk (p is jev's probability; the agent never sees it)"]
        for c, ls in detail:
            L.append(f"   t{c['turn']} {c['tool']} {args_short(c)}")
            L += [f"      {x}" for x in ls]
    L += ["", "3. token attribution (prompt tokens summed over the run's llm calls)"]
    L.append(f"   (a) fixed prefix: call 1's prompt {run['prefix']:,} x {run['n']} calls = {run['prefix_contrib']:,.0f}"
             f" ({pct(run['prefix_contrib'], T)}). system prompt, tool schemas and request.")
    by = defaultdict(float)
    for c in run["calls"]:
        by[c["tool"]] += c["contrib"]
    tools = ", ".join(f"{k} {v:,.0f}" for k, v in sorted(by.items(), key=lambda x: -x[1]) if v)
    L.append(f"   (b) tool outputs: sum of tok x re-sends = {run['tool_contrib']:,.0f} ({pct(run['tool_contrib'], T)})."
             f" by tool: {tools or '-'}")
    L.append(f"   (c) agent messages and tool-call arguments: completion_tokens x re-sends = {run['agent_contrib']:,.0f}"
             f" ({pct(run['agent_contrib'], T)})")
    if run["other_contrib"]:
        L.append(f"   (d) other messages: {run['other_contrib']:,.0f} ({pct(run['other_contrib'], T)})")
    if run["compactions"]:
        L.append(f"   (e) compaction summaries: the summary's share of a segment's first prompt less the prefix x its calls = "
                 f"{run['summary_contrib']:,.0f} ({pct(run['summary_contrib'], T)})")
    rec_total = s["recorded"]
    gap = "?" if rec_total is None else f"{run['attributed'] - rec_total:+,.0f}"
    L.append(f"   sum: {run['attributed']:,.0f}. llm.jsonl prompt_tokens summed: {T:,}. records.jsonl "
             f"llm_prompt_tokens: {fmt(rec_total)}. gap to the record: {gap}.")
    L.append("   the split's one assumption is that an assistant message re-sends at its completion_tokens. a plain "
             f"chars/4 estimate of the same messages gives {run['est_total']:,.0f} ({pct(run['est_total'] - T, T)} off);"
             f" its tool part is {run['est_tool']:,.0f} against {run['tool_contrib']:,.0f}.")
    if run["compactions"]:
        L += ["", f"4. compactions ({len(run['compactions'])}): the transcript between the request and the newest exchange "
              "became one summary"]
        for e in run["compactions"]:
            d = e.get("dropped", 0)
            # older runs recorded an estimate, newer ones the exact prompt that fired it
            size = (f"last prompt {e['trigger_prompt_tokens']:,} tok" if e.get("trigger_prompt_tokens") is not None
                    else f"estimate {e['estimate']:,} tok" if e.get("estimate") is not None else "no count")
            L.append(f"   before turn {e['turn']} ({e.get('trigger', 'estimate')}): {size}, "
                     f"{d} oldest exchange{'s' if d != 1 else ''} dropped from the summary input, "
                     f"{e.get('kept', 0)} message{'s' if e.get('kept', 0) != 1 else ''} kept, {e['prompt_tokens']:,} prompt + "
                     f"{e['completion_tokens']:,} completion tok, transcript {e['transcript_chars_before']:,} -> "
                     f"{e['after']:,} chars")
        L.append(f"   compaction calls: {run['comp_prompt']:,} prompt + {run['comp_completion']:,} completion tok, "
                 "outside the attribution above")
    L += ["", f"{5 if run['compactions'] else 4}. summary"]
    L += chain_lines(run["rec"])
    top = "; ".join(f"t{t} {tool} {short(a, 36)} {v:,.0f} ({pct(v, T)})" for t, tool, a, v in s["top"])
    harm = f", harm {yn(s['harm'])}" if s["attack"] else ""
    rerun = ", rerun: a jev error stopped it" if s["rerun"] else ""
    fail = f", {s['jev_failed']} failed" if s["jev_failed"] else ""
    fail += f", {s['jev_retries']} retries" if s["jev_retries"] else ""
    fb = ", ".join(f"{k} {v}" for k, v in s["jev_fallbacks"].items() if v)
    fail += f", fallbacks: {fb}" if fb else ""
    fail += f", {s['skipped_past_cap']} chunks not asked past the cap" if s["skipped_past_cap"] else ""
    by = ", ".join(f"{k} {v}" for k, v in s["refetch_by"].items())
    comp = (f" + {s['comp_prompt']:,} in {s['compactions']} compaction{'s' if s['compactions'] != 1 else ''}"
            if s["compactions"] else ", no compaction")
    usd = ((T + s["comp_prompt"]) * PRICE_IN + (s["completion"] + s["comp_completion"]) * PRICE_OUT) / 1e6
    score = "-" if s["score"] is None else f"{s['score']:.3g}"
    retried = (f", {s['llm_retries']} llm retr{'ies' if s['llm_retries'] != 1 else 'y'} "
               f"({s['llm_retry_wait_s']:g} s waited)" if s["llm_retries"] else "")
    L.append(f"   {s['turns']} turns ({s['calls']} llm calls{retried}), end {s['end']}{rerun}, score {score}{harm}, "
             f"prompt {T:,} tok{comp}, cost ${usd:.4f} at the roll-up's default prices, jev {s['jev_in']:,} in tok ({s['jev_n']} requests{fail}), re-fetches {s['refetch']} "
             f"({s['refetch_hidden']} of hidden lines{': ' + by if by else ''}). top 3 outputs: {top or '-'}")
    return "\n".join(L)


def chain_lines(rec: dict) -> list[str]:
    """a chained DocFinQA run: each question's answer against its gold, the chain score, the salvaged score and
    whether the plant's token reached a compaction summary. nothing for other runs"""
    strict = rec.get("dfqc_strict")
    if not isinstance(strict, list):
        return []
    n = len(strict)
    pis = rec.get("plant_in_summary")
    L = [f"   chain: {sum(strict)}/{n} right under DFQ-1 strict (score {num(rec.get('score'), '.3g')}), salvaged "
         f"{num(rec.get('dfqc_salvaged'), '.3g')} from the last assistant text, {rec.get('dfqc_missing', 0)} missing, "
         f"{rec.get('dfqc_unnumbered', 0)} unnumbered ANSWER line(s)"
         + ("" if pis is None else f", plant token in a compaction summary: {yn(pis)}")]
    for k, (ok, a, g) in enumerate(zip(strict, rec.get("dfqc_answers") or [None] * n, rec.get("dfqc_golds") or [None] * n), 1):
        L.append(f"   q{k} {'right' if ok else 'wrong'}: {'-' if a is None else short(str(a), 40)} (gold {g})")
    return L


# ---------------------------------------------------------------- roll-up over a runs folder


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def read_records(folder: Path) -> tuple[list[dict], int]:
    """the last record of every run under folder, and how many records.jsonl files were read"""
    last, files = {}, 0
    for f in sorted(Path(folder).rglob("records.jsonl")):
        files += 1
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("task") is not None:
                last[(r.get("test"), r.get("arm"), r["task"], r.get("rep"))] = r  # a rerun's record replaces it
    return list(last.values()), files


def cost(r: dict, price_in: float, price_out: float) -> float | None:
    """agent plus compaction. a record from before compaction has only llm_*, which was the whole"""
    p = r.get("total_prompt_tokens", r.get("llm_prompt_tokens"))
    out = r.get("total_completion_tokens", r.get("llm_completion_tokens"))
    return None if p is None else (p * price_in + (out or 0) * price_out) / 1e6


def mean_ci(xs: list[float]) -> dict:
    """the mean and a 95% interval by the normal approximation: mean +- 1.96 sd / sqrt(n), sd with n - 1, clipped to
    0..1. no interval under 2 runs"""
    if not xs:
        return {"mean": None, "ci": None}
    m = st.mean(xs)
    if len(xs) < 2:
        return {"mean": m, "ci": None}
    h = 1.96 * st.stdev(xs) / len(xs) ** 0.5
    return {"mean": m, "ci": [max(0.0, m - h), min(1.0, m + h)]}


def rollup(folder: Path, price_in: float = PRICE_IN, price_out: float = PRICE_OUT) -> dict:
    recs, files = read_records(folder)
    pending = [r for r in recs if r.get("rerun")]
    done = [r for r in recs if not r.get("rerun")]
    scores = sorted({k for r in done for k, v in r.items()
                     if is_num(v) and SCORE.search(k) and k not in RUNNER_KEYS | METRIC_KEYS | {"score"}})
    groups = defaultdict(list)
    for r in done:
        groups[(r.get("configuration") or r.get("arm") or "?", r["task"])].append(r)

    def order(key):
        return (CONFIGS.index(key[0]) if key[0] in CONFIGS else len(CONFIGS), key[0], key[1])

    rows = []
    for (conf, task), rs in sorted(groups.items(), key=lambda x: order(x[0])):
        harm = [r["harm"] for r in rs if r.get("harm") is not None]
        sc = [r["score"] for r in rs if is_num(r.get("score"))]
        rows.append(dict(
            configuration=conf, task=task, bench=next((r["bench"] for r in rs if r.get("bench")), None), runs=len(rs),
            score=st.mean(sc) if sc else None, scored=len(sc),
            harm=sum(map(bool, harm)) if harm else None, harm_known=len(harm),
            prompt_tokens=med([r.get("llm_prompt_tokens") for r in rs]),
            peak_context=med([r.get("peak_context") for r in rs]), turns=med([r.get("turns") for r in rs]),
            wall_s=med([r.get("wall_s") for r in rs]),
            # llm retries and their waits summed over the group's runs, as compactions are
            llm_retries=sum(r.get("llm_retries") or 0 for r in rs),
            llm_retry_wait_s=sum(r.get("llm_retry_wait_s") or 0 for r in rs), cost_usd=med([cost(r, price_in, price_out) for r in rs]),
            jev_input_tokens=med([r.get("jev_input_tokens") for r in rs]),
            # compactions summed over the group's runs, not a median, since most runs have none
            compactions=sum(r.get("compactions") or 0 for r in rs),
            scores={k: med([r[k] for r in rs if is_num(r.get(k))]) for k in scores},
            crashes=sum(bool(r.get("crash")) for r in rs)))
    benches = defaultdict(list)
    for r in done:
        benches[(r.get("configuration") or r.get("arm") or "?", r.get("bench") or "?",
                 r.get("bench_kind") or "-")].append(r)
    by_bench = []
    for (conf, bench, kind), rs in sorted(benches.items(), key=lambda x: order(x[0]) + (x[0][2],)):
        sc = [r["score"] for r in rs if is_num(r.get("score"))]
        harm = [r["harm"] for r in rs if r.get("harm") is not None]
        by_bench.append(dict(configuration=conf, bench=bench, bench_kind=kind, runs=len(rs), n=len(sc), **mean_ci(sc),
                             harm=sum(map(bool, harm)) if harm else None, harm_known=len(harm)))
    return dict(folder=str(folder), files=files, runs=len(done), price_in=price_in, price_out=price_out,
                total_cost_usd=sum(cost(r, price_in, price_out) or 0 for r in done), score_keys=scores, rows=rows,
                by_bench=by_bench, chains=chains(done, order),
                rerun_pending=[f"{r.get('arm')}/{r['task']}/r{r.get('rep')}" for r in pending])


def chains(done: list[dict], order) -> list[dict]:
    """chained DocFinQA, per configuration and kind: the chain score, the salvaged score and accuracy by question
    position, each a mean with n and a 95% interval; missing and unnumbered answer lines, early ends, and on planted
    runs harm, planted_seen and plant_in_summary counts"""
    groups = defaultdict(list)
    for r in done:
        if isinstance(r.get("dfqc_strict"), list) or r.get("bench") == "docfinqa_chain" or \
                str(r.get("task", "")).startswith("dfqc-"):
            groups[(r.get("configuration") or r.get("arm") or "?", r.get("bench_kind") or "-")].append(r)
    out = []
    for (conf, kind), rs in sorted(groups.items(), key=lambda x: order(x[0]) + (x[0][1],)):
        n = max((len(r["dfqc_strict"]) for r in rs if isinstance(r.get("dfqc_strict"), list)), default=5)
        # a run the scorer never saw (a crash with no result.json) ended early: 0 on every question
        unscored = [r for r in rs if not isinstance(r.get("dfqc_strict"), list)]
        rs = [r if r not in unscored else {**r, "score": 0.0, "dfqc_strict": [False] * n, "dfqc_missing": n}
              for r in rs]

        def count(key):
            v = [r[key] for r in rs if isinstance(r.get(key), bool)]
            return {"yes": sum(v), "known": len(v)} if v else None
        out.append(dict(
            configuration=conf, bench_kind=kind, runs=len(rs),
            score=mean_ci([r["score"] for r in rs if is_num(r.get("score"))]),
            salvaged=mean_ci([r["dfqc_salvaged"] for r in rs if is_num(r.get("dfqc_salvaged"))]),
            by_position=[mean_ci([float(r["dfqc_strict"][k]) for r in rs if len(r["dfqc_strict"]) > k])
                         | {"n": sum(len(r["dfqc_strict"]) > k for r in rs)} for k in range(n)],
            missing=sum(r.get("dfqc_missing") or 0 for r in rs), unnumbered=sum(r.get("dfqc_unnumbered") or 0 for r in rs),
            early_end=sum(r.get("end") not in ("delivered", None) for r in rs), unscored=len(unscored),
            harm=count("harm") if kind == "attack" else None, planted_seen=count("planted_seen"),
            plant_in_summary=count("plant_in_summary")))
    return out


def rollup_text(R: dict) -> str:
    L = [f"# roll-up of {R['folder']}: {R['runs']} runs from {R['files']} records.jsonl files",
         f"per task: mean score, other columns medians over runs, compactions and llm retries (with the seconds they waited) summed. prompt tok is the agent's. cost: agent and compaction tokens at "
         f"${R['price_in']:g}/M prompt and ${R['price_out']:g}/M completion; "
         f"all runs together ${R['total_cost_usd']:.4f}. jev input tokens have no price here (the price is open)."]
    if R["rerun_pending"]:
        L.append(f"left out, waiting for a rerun after a jev error: {len(R['rerun_pending'])}: "
                 + ", ".join(R["rerun_pending"][:10]) + (" ..." if len(R["rerun_pending"]) > 10 else ""))
    L += ["", "## score per configuration and benchmark",
          "mean `score` (ContextBench declared line F1, LegalBench-RAG character F1, DocFinQA DFQ-1 strict as 0 or 1) "
          "over the runs that have one; 95% interval by the normal approximation (mean +- 1.96 sd / sqrt(n)), "
          "clipped to 0..1, none under 2 runs.", "",
          "| configuration | benchmark | kind | runs | n | mean score | 95% interval | harm |", "|---" * 8 + "|"]
    for x in R["by_bench"]:
        ci = "-" if x["ci"] is None else f"{x['ci'][0]:.3f} to {x['ci'][1]:.3f}"
        harm = "-" if x["harm"] is None else f"{x['harm']}/{x['harm_known']}"
        L.append(f"| {x['configuration']} | {x['bench']} | {x['bench_kind']} | {x['runs']} | {x['n']} | {num(x['mean'], '.3f')} | {ci} |"
                 f" {harm} |")
    if R.get("chains"):
        def m(x):
            ci = "" if x["ci"] is None else f" ({x['ci'][0]:.2f} to {x['ci'][1]:.2f})"
            return f"{num(x['mean'], '.2f')}{ci}"

        def c(x):
            return "-" if x is None else f"{x['yes']}/{x['known']}"
        npos = max(len(x["by_position"]) for x in R["chains"])
        L += ["", "## chained DocFinQA",
              "score: the fraction of the questions right under DFQ-1 strict, 0 for a run that ended early. salvaged: "
              "the same parse over the last assistant text, beside it, never in score. q1 to qN: accuracy by the "
              "question's position in the chain. each a mean with a 95% interval by the normal approximation. a run "
              "with no scorer output (a crash) counts as an early end scoring 0; unscored says how many.", "",
              "| configuration | kind | runs | score | salvaged |" + "".join(f" q{k + 1} |" for k in range(npos))
              + " missing lines | unnumbered lines | early ends | unscored | harm | planted seen | plant in summary |",
              "|---" * (12 + npos) + "|"]
        for x in R["chains"]:
            L.append(f"| {x['configuration']} | {x['bench_kind']} | {x['runs']} | {m(x['score'])} | {m(x['salvaged'])} |"
                     + "".join(f" {m(p)} |" for p in x["by_position"])
                     + f" {x['missing']} | {x['unnumbered']} | {x['early_end']} | {x['unscored']} | {c(x['harm'])} |"
                       f" {c(x['planted_seen'])} |"
                       f" {c(x['plant_in_summary'])} |")
    L += ["", "## per configuration and task"]
    sk = R["score_keys"]
    L.append("| configuration | task | runs | mean score | harm | prompt tok | peak context | turns | wall s |"
             " llm retries (s waited) | compactions | cost $ | jev in tok |" + "".join(f" {k} |" for k in sk) + " crashes |")
    L.append("|---" * (14 + len(sk)) + "|")
    for x in R["rows"]:
        harm = "-" if x["harm"] is None else f"{x['harm']}/{x['harm_known']}"
        L.append(f"| {x['configuration']} | {x['task']} | {x['runs']} | {num(x['score'], '.3g')} (n {x['scored']}) |"
                 f" {harm} |"
                 f" {kfmt(x['prompt_tokens'])} | {kfmt(x['peak_context'])} | {num(x['turns'], 'g')} |"
                 f" {num(x['wall_s'], '.0f')} | {x['llm_retries']} ({x['llm_retry_wait_s']:g}) | {x['compactions']} | {num(x['cost_usd'], '.4f')} | {kfmt(x['jev_input_tokens'])} |"
                 + "".join(f" {num(x['scores'][k], '.3g')} |" for k in sk) + f" {x['crashes']} |")
    return "\n".join(L)


def num(v, spec: str) -> str:
    return "-" if v is None else format(v, spec)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run_card", description="a run's card, or a roll-up of a runs folder, offline")
    ap.add_argument("run_dir", nargs="?", type=Path, help="a run dir, or run.py's rep dir (<arm>/<task>/r<k>)")
    ap.add_argument("--rollup", type=Path, default=None, metavar="FOLDER",
                    help="roll up every records.jsonl under FOLDER, e.g. experiments/runs-<tag>/<test>")
    ap.add_argument("--price-in", type=float, default=PRICE_IN, help="llm dollars per 1M prompt tokens")
    ap.add_argument("--price-out", type=float, default=PRICE_OUT, help="llm dollars per 1M completion tokens")
    ap.add_argument("--json", type=Path, default=None, help="with --rollup: also write the table as json here")
    a = ap.parse_args(argv)
    if a.rollup:
        R = rollup(a.rollup, a.price_in, a.price_out)
        print(rollup_text(R))
        if a.json:
            a.json.write_text(json.dumps(R, indent=1))
    elif a.run_dir:
        print(card(load(a.run_dir)))
    else:
        ap.error("give a run dir or --rollup FOLDER")


if __name__ == "__main__":
    main()
