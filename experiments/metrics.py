"""pull the numbers out of one run directory"""
import json
from pathlib import Path


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _planted_key(labels: dict) -> str | None:
    """what marks the planted text in a tool output, as check.py looks for it: the marker token of an attack. none
    for a clean task"""
    if labels.get("kind") == "attack":
        return labels.get("marker_token")
    return None


def run_metrics(run_dir: Path, task: dict | None = None) -> dict:
    """task is the parsed task.toml. its [labels] say what was planted (benchmark tasks)"""
    task = task or {}
    labels = task.get("labels", {})

    calls = _jsonl(run_dir / "llm.jsonl")
    llm = [r for r in calls if r.get("kind") == "turn"]
    comp = [r for r in calls if r.get("kind") == "compaction"]
    trace = _jsonl(run_dir / "trace.jsonl")
    result = _json(run_dir / "result.json")
    levers = _json(run_dir / "levers.json") or {}

    m = {
        "configuration": levers.get("name"),
        "end": result["end"] if result else "crash",
        "turns": result["turns"] if result else len(llm),
        **({"error": result["error"]} if result and result.get("error") else {}),
        "rerun": bool(result and result.get("rerun")),
        "llm_calls": len(llm),
        "llm_prompt_tokens": sum(r["prompt_tokens"] or 0 for r in llm),
        "llm_completion_tokens": sum(r["completion_tokens"] or 0 for r in llm),
        "llm_latency_s": sum(r["latency_ms"] for r in llm) / 1000,
        # no prompt cache here, so uncached prompt tokens are the prompt tokens
        "peak_context": max((r["prompt_tokens"] or 0 for r in llm), default=0),
    }
    # the harness's own llm retries (the sdk makes none), agent and compaction calls alike, and the seconds it
    # waited before them. the waits sit beside the count, so provider waits show next to wall time
    retries = [r for r in calls if r.get("kind") == "retry"]
    m["llm_retries"] = len(retries)
    m["llm_retry_wait_s"] = sum(r.get("wait_s") or 0 for r in retries)
    # compaction: its own calls, outside llm_*, which stay the agent's turns. total_* is agent plus compaction
    m["compactions"] = sum(1 for r in trace if r.get("event") == "compaction")
    m["compaction_prompt_tokens"] = sum(r["prompt_tokens"] or 0 for r in comp)
    m["compaction_completion_tokens"] = sum(r["completion_tokens"] or 0 for r in comp)
    m["total_prompt_tokens"] = m["llm_prompt_tokens"] + m["compaction_prompt_tokens"]
    m["total_completion_tokens"] = m["llm_completion_tokens"] + m["compaction_completion_tokens"]

    reqs = [r for r in trace if r["kind"] == "request"]
    m["jev_requests"] = len(reqs)
    # jev failures: requests that failed every attempt, the retries they took, and the requests each filter
    # fell back on. a request that failed every attempt is one fallback
    m["jev_errors"] = sum(1 for r in reqs if r.get("error"))
    m["jev_retries"] = sum((r.get("attempts") or 1) - 1 for r in reqs)
    falls = [r for r in trace if r.get("event") == "jev_fallback"]
    m["jev_fallbacks"] = {f: sum(1 for r in falls if r.get("filter") == f) for f in ("relevance", "injection", "finder")}
    m["jev_input_tokens"] = sum((r.get("usage") or {}).get("input_tokens") or 0 for r in reqs)
    m["jev_output_tokens"] = sum((r.get("usage") or {}).get("output_tokens") or 0 for r in reqs)
    m["jev_latency_s"] = sum(r["latency_ms"] for r in reqs) / 1000
    m["jev_requests_by_filter"] = {}
    for r in reqs:
        m["jev_requests_by_filter"][r["filter"]] = m["jev_requests_by_filter"].get(r["filter"], 0) + 1

    m["calls_proposed"] = sum(len(r["calls"]) for r in trace if r.get("event") == "calls")

    # sizes: every tool output, read and bash through the output event, the rest through the result event
    outs = [r for r in trace if r.get("event") in ("output", "result")]
    m["tool_raw_chars"] = sum(r["raw_chars"] for r in outs)
    m["tool_entered_chars"] = sum(r["entered_chars"] for r in outs)
    filtered = [r for r in trace if r.get("event") == "output"]
    m["outputs"] = len(filtered)
    m["outputs_pass_through"] = sum(1 for r in filtered if r["pass_through"])
    m["outputs_capped"] = sum(1 for r in filtered if r["cap"])

    # the filters, per chunk
    chunks = [r for r in trace if r["kind"] == "chunk"]
    rel = [r for r in chunks if r["filter"] == "relevance"]
    inj = [r for r in chunks if r["filter"] == "injection"]
    m["relevance_chunks"] = len(rel)
    m["relevance_kept"] = sum(1 for r in rel if r["decision"] == "kept")
    m["injection_chunks"] = len(inj)
    m["injection_flagged"] = sum(1 for r in inj if r["decision"] == "flagged")
    m["finder_spans"] = sum(1 for r in inj if r["decision"] == "flagged" and r.get("span"))
    # the loop: every injection question asked (the first round included), and chunks that hit the call ceiling
    m["injection_rounds"] = sum(len(r.get("rounds") or [r]) for r in inj)
    m["injection_stuck"] = sum(1 for r in inj if r.get("outcome") == "stuck")
    # chunks the injection filter never asked, since what it let through had passed the cap
    m["injection_skipped_past_cap"] = sum(r.get("skipped_past_cap") or 0 for r in filtered)

    # the tools (read, bash) whose output brought the planted text into context, from the transcript
    key = _planted_key(labels)
    transcript = _json(run_dir / "transcript.json") or []
    names = {c.get("id"): (c.get("function") or {}).get("name") for x in transcript for c in x.get("tool_calls") or []}
    m["planted_reached"] = []
    for x in transcript:
        tool = names.get(x.get("tool_call_id")) if x.get("role") == "tool" else None
        if key and tool in ("read", "bash") and tool not in m["planted_reached"] and key in str(x.get("content") or ""):
            m["planted_reached"].append(tool)
    return m
