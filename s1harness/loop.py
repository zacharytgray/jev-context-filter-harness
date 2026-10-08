import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import messages as M
from . import compaction as C
from .jev import Jev, JevError, Trace
from .levers import Levers
from .llm import LLM, ContextOverflow
from .pipeline import (MAX_TURNS, PASS_THROUGH_CHARS, bash_pipeline, constants, needs_question, read_command,
                       read_pipeline)
from .tools import schemas, Tools


@dataclass
class RunResult:
    final_message: str
    end: str  # delivered, max turns, jev error. a crash writes overflow or error
    turns: int
    run_dir: Path
    rerun: bool = False  # a jev error reached the loop: the run is rerun and replaced, never reported


def _parse(call: dict, names: set[str], relevance: bool) -> tuple[str, str, dict, str | None]:
    """(call id, tool, args, error or None)"""
    name, raw = call["function"]["name"], call["function"]["arguments"]
    try:
        args, err = json.loads(raw or "{}"), None
    except json.JSONDecodeError:
        args, err = {}, "Error: the arguments were not valid JSON."
    if not isinstance(args, dict):
        args, err = {}, "Error: the arguments must be a JSON object."
    if name not in names:
        err = f"Error: there is no tool named {name}."
    elif err is None and relevance and name in ("read", "bash") and not str(args.get("question") or "").strip():
        err = needs_question(name)
    return call.get("id", ""), name, args, err


def request_note(request: str, summary: str) -> str:
    """the second message after a compaction: the request and the summary in one user message"""
    return request + "\n\n" + C.SUMMARY_HEAD + summary


def run(request: str, workdir: Path, levers: Levers, llm: LLM, jev: Jev, trace: Trace,
        context_window: int = C.CONTEXT_WINDOW,
        max_output: int | None = None, max_turns: int = MAX_TURNS,
        compaction_budget: int = C.COMPACTION_BUDGET) -> RunResult:
    run_dir = trace.run_dir
    limit = C.budget(context_window, max_output, compaction_budget)
    # the window and the compaction budget ride in levers.json, next to the filters' constants. budget is
    # the one the trigger uses, compaction_budget the run's own (default 32,768). max_turns is the run's own limit
    lv = {**levers.to_dict(), "constants": {**constants(), "max_turns": max_turns},
          "compaction": {"context_window": context_window, "max_output": max_output, "margin": C.MARGIN,
                         "compaction_budget": compaction_budget, "budget": limit}}
    tools = Tools(workdir)
    tool_schemas = schemas(levers.relevance)
    names = {t["function"]["name"] for t in tool_schemas}
    (run_dir / "levers.json").write_text(json.dumps(lv, indent=1))
    (run_dir / "tools.json").write_text(json.dumps(tool_schemas, indent=1))
    msgs = [{"role": "system", "content": M.system_prompt()}, {"role": "user", "content": request}]
    # transcript.json keeps every message that entered context, a summary before the exchange kept with it
    # msgs is what the model sees now; it goes to context.json once a compaction has changed it
    history = list(msgs)
    last = {"tokens": None}  # the last agent call's exact prompt tokens. none before the first, after a compaction
    compactions = 0
    summaries = []  # each compaction's summary text
    switches = {"relevance": levers.relevance, "injection": levers.injection}

    def finish(text: str, end: str, turn: int, error: str | None = None, rerun: bool = False) -> RunResult:
        err = {"error": error} if error else {}
        trace.event(event="end", end=end, **err)
        (run_dir / "transcript.json").write_text(json.dumps(history, indent=1))
        if compactions:
            (run_dir / "context.json").write_text(json.dumps(msgs, indent=1))
        (run_dir / "result.json").write_text(json.dumps(
            {"request": request, "final_message": text, "end": end, "turns": turn, "max_turns": max_turns, **err,
             **({"rerun": True} if rerun else {})}, indent=1))
        return RunResult(text, end, turn, run_dir, rerun)

    def output(cid: str, name: str, args: dict) -> str:
        """read and bash: the tool, then the configuration's filters. one output event per call"""
        question = str(args.get("question") or "").strip()
        # bash drops a question it does not take, read only when it takes one
        plain = {k: v for k, v in args.items() if k != "question"} if name == "bash" or levers.relevance else args
        t0 = time.monotonic()
        res = tools.run(name, plain)
        tool_ms = round((time.monotonic() - t0) * 1000)
        where = {"path": args.get("path")} if name == "read" else {"command": args.get("command")}
        sent, tokens = jev.requests, jev.input_tokens
        if not res.ok:  # an error from the tool enters as it is, unfiltered, logged with ok false
            n = len(res.text)
            text, records = res.text, []
            info = {"raw_chars": n, "raw_lines": len(res.text.split("\n")), "pass_through": n <= PASS_THROUGH_CHARS,
                    "after_relevance": n, "after_injection": n, "after_cap": n, "entered_chars": n, "cap": None,
                    "skipped_past_cap": 0, "jev_ms": 0, "shown": [], "hidden": []}
        elif name == "read":
            rel = str(tools._resolve(res.path).relative_to(workdir.resolve()))
            ranged = bool(plain.get("start_line") or plain.get("end_line"))
            # jev's relevance state names the call, with the agent's own path and range
            command = read_command(plain.get("path"), plain.get("start_line"), plain.get("end_line"))
            text, info, records = read_pipeline(jev, request, question, res, rel, ranged, command, call_id=cid,
                                                **switches)
        else:
            text, info, records = bash_pipeline(jev, request, question, res.text, str(plain.get("command") or ""),
                                                workdir, call_id=cid, **switches)
            # harness text after the filters, never filtered or counted as output
            text = "\n".join(x for x in (text, f"(exit code {res.exit_code})") if x)
            info["entered_chars"] = len(text)
        for r in records:
            trace.chunk(cid, r)
        trace.event(event="output", call_id=cid, tool=name, **where, question=question or None, ok=res.ok,
                    **{k: info[k] for k in ("raw_chars", "raw_lines", "pass_through", "after_relevance",
                                            "after_injection", "after_cap", "entered_chars", "cap",
                                            "skipped_past_cap")},
                    tool_ms=tool_ms, jev_ms=info["jev_ms"], jev_requests=jev.requests - sent,
                    jev_input_tokens=jev.input_tokens - tokens, shown=info["shown"], hidden=info["hidden"])
        return text

    def add(m: dict):
        msgs.append(m)
        history.append(m)

    def compact(turn: int, trigger: str) -> bool:
        """summarize what lies between the request and the newest exchange. false when there is nothing to"""
        nonlocal compactions
        fired = last["tokens"] if trigger == "exact" else None
        k = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=None)
        # an earlier summary alone is not summarized again
        if k is None or k <= 2:
            return False
        newest = msgs[k:]  # the agent's last message, its calls and all their results, kept verbatim
        before = C.chars(msgs)
        # the request fills {request}, so {older} starts after it, with the last summary first
        earlier = [{"role": "user", "content": C.SUMMARY_HEAD + summaries[-1]}] if summaries else []
        older = C.exchanges(msgs[2:k])
        summary, dropped, pt, ct = C.summarize(llm, request, earlier, older)
        # the request and the summary share one user message, so roles still alternate
        note = {"role": "user", "content": request_note(request, summary)}
        msgs[1:k] = [note]
        summaries.append(summary)
        # transcript.json holds the note where it sits in context, before the exchange kept with it
        at = len(history) - len(newest)
        history.insert(at, note)
        # after a compaction the trigger waits for the next agent call's own count
        last["tokens"] = None
        compactions += 1
        # trigger, trigger_prompt_tokens, dropped, kept, summary_chars, and at (the note's index in
        # transcript.json). trigger_prompt_tokens is the exact count that fired it, none for a rejection
        trace.event(event="compaction", turn=turn, trigger=trigger,
                    trigger_prompt_tokens=fired, dropped=dropped,
                    kept=len(newest), prompt_tokens=pt, completion_tokens=ct, transcript_chars_before=before,
                    after=C.chars(msgs), summary_chars=len(summary), at=at)
        return True

    def agent_call(turn: int) -> dict:
        # the trigger is the provider's exact prompt_tokens. the last agent call's prompt passed
        # the budget, so compaction runs before this one. no character estimate
        if last["tokens"] is not None and last["tokens"] > limit:
            compact(turn, "exact")
        try:
            msg, tokens = llm.chat(msgs, tool_schemas)
        except ContextOverflow:
            # the provider said no: compact once and retry. a second no ends the run as overflow
            if not compact(turn, "rejected"):
                raise
            msg, tokens = llm.chat(msgs, tool_schemas)
        # only agent calls count, never a compaction call. a call with no usage (llm.chat gives 0) leaves
        # no count, so the next call does not compact; the rejection path stays the backstop
        last["tokens"] = tokens or None
        return msg

    try:
        for turn in range(1, max_turns + 1):
            trace.turn = turn
            msg = agent_call(turn)
            add(msg)
            calls = msg.get("tool_calls") or []
            if not calls:  # a reply with no tool call ends the run
                return finish(msg["content"], "delivered", turn)
            parsed = [_parse(c, names, levers.relevance) for c in calls]
            trace.event(event="calls", calls=[{"call_id": cid, "tool": name, "args": a} for cid, name, a, _ in parsed])
            for cid, name, args, err in parsed:
                if err:  # a call that never ran logs a result event too
                    text = err
                    trace.event(event="result", call_id=cid, tool=name, raw_chars=len(text), entered_chars=len(text),
                                error=True)
                elif name in ("read", "bash"):
                    text = output(cid, name, args)
                else:
                    text = tools.run(name, args).text
                    trace.event(event="result", call_id=cid, tool=name, raw_chars=len(text), entered_chars=len(text))
                # appended per call, so a jev error keeps the turn's earlier results in the transcript
                add({"role": "tool", "tool_call_id": cid, "content": text})
    except JevError as e:
        # a safety net: the filters fall back open on a failed request, so no JevError should get here.
        # if one does, the run stops, keeps its records, and is rerun
        trace.event(event="jev_error", call_id=e.call_id, filter=e.filter, error=str(e))
        return finish("", "jev error", trace.turn, f"JevError: {e}", rerun=True)
    except Exception as e:
        # a crashed run keeps its records, then fails as before
        finish("", "overflow" if isinstance(e, ContextOverflow) else "error", trace.turn, f"{type(e).__name__}: {e}")
        raise

    return finish("", "max turns", max_turns)
