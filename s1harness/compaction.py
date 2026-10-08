"""standard compaction, in every configuration: once an agent call's exact prompt_tokens pass the budget, one llm
call before the next summarizes the work between the request and the newest exchange. the system prompt and the
newest exchange stay verbatim"""
import json

from .llm import LLM, ContextOverflow

# the summary wording, fixed
SUMMARY_SYSTEM = "You summarize the earlier part of an agent's work so the agent can continue without it."
SUMMARY_USER = """Here is the earlier part of the agent's work on this request:

{request}

{older}

Summarize it for the agent: what it has learned, what it has changed, what it tried that failed, and what is left to do. Be concise and keep exact file names, function names, and error messages that matter."""
SUMMARY_HEAD = "[summary of earlier work]\n"

CONTEXT_WINDOW = 262144  # gemma 4 31b
COMPACTION_BUDGET = 32768  # compaction triggers past this many tokens. the window stays as it is
# a run can set its own budget (16,384 for the 16K run set). this is the default
MARGIN = 4096


def chars(msgs: list[dict]) -> int:
    """the compaction event's sizes, in characters. never the trigger"""
    return sum(len(json.dumps(m)) for m in msgs)


def budget(window: int, max_output: int | None, compaction_budget: int = COMPACTION_BUDGET) -> int:
    # no --max-output reserves nothing for the reply
    # the smaller of the run's budget and the window's room, so a smaller real window still protects
    return min(compaction_budget, window - (max_output or 0) - MARGIN)


def render(msgs: list[dict]) -> str:
    """plain text of the messages: role, tool name and args, content"""
    # this layout, one block per message or call, blank lines between
    names = {c.get("id"): c["function"]["name"] for m in msgs for c in m.get("tool_calls") or []}
    out = []
    for m in msgs:
        if m["role"] == "assistant":
            if m.get("content"):
                out.append("assistant:\n" + m["content"])
            for c in m.get("tool_calls") or []:
                out.append(f"assistant calls {c['function']['name']}: {c['function']['arguments']}")
        elif m["role"] == "tool":
            out.append(f"tool {names.get(m.get('tool_call_id'), '?')}:\n{m['content']}")
        else:
            out.append(f"{m['role']}:\n{m['content']}")
    return "\n\n".join(out)


def exchanges(msgs: list[dict]) -> list[list[dict]]:
    """messages cut into exchanges: an assistant message and the tool results after it"""
    out = []
    for m in msgs:
        if m["role"] == "assistant" or not out:
            out.append([])
        out[-1].append(m)
    return out


def summarize(llm: LLM, request: str, earlier: list[dict], older: list[list[dict]]) -> tuple[str, int, int, int]:
    """one call over the earlier summary (a message, or none) and the older exchanges. a call the provider rejects
    for context length is retried without the oldest exchange, one at a time. none left to drop raises
    ContextOverflow. returns (summary, exchanges dropped, prompt tokens, completion tokens)"""
    dropped = 0
    while True:
        older_text = render(earlier + [m for ex in older[dropped:] for m in ex])
        try:
            msg, p = llm.chat([{"role": "system", "content": SUMMARY_SYSTEM},
                               {"role": "user", "content": SUMMARY_USER.format(request=request, older=older_text)}],
                              kind="compaction")
        except ContextOverflow:
            if dropped >= len(older):  # nothing droppable left
                raise
            dropped += 1
            continue
        return msg["content"].strip(), dropped, p, llm.completion_tokens or 0
