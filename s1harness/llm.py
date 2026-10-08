import time

from openai import APIConnectionError, APIStatusError, OpenAI

from .jev import Trace


# the sdk retries nothing, so every retry is the harness's own and is logged in llm.jsonl.
# a hosted provider can be busy for minutes: wait it out instead of losing the run. these waits cover every error
# retryable() names
HOSTED_WAITS = (15, 30, 60, 60, 120, 120, 240)
# a local server gets 2 retries, after 1 and 2 s, as the sdk would give it
LOCAL_WAITS = (1, 2)


class ContextOverflow(Exception):
    pass


def retryable(e: Exception) -> bool:
    """what the sdk retried: a connection error or a timeout, 408, 409, 429, and every 5xx"""
    # the same set as the sdk's, 408 and 409 included
    if isinstance(e, APIConnectionError):  # a timeout is one too
        return True
    return isinstance(e, APIStatusError) and (e.status_code in (408, 409, 429) or e.status_code >= 500)


def too_long(e: APIStatusError) -> bool:
    """a provider's rejection of a prompt longer than the model's context"""
    # a 400 or 413 whose message names the context or a too long prompt
    text = str(e).lower()
    return e.status_code in (400, 413) and any(w in text for w in ("context", "too long", "too many tokens"))


class LLM:
    def __init__(self, base_url: str, model: str, trace: Trace, api_key: str = "none", provider: str | None = None,
                 max_context: int | None = None, reasoning_effort: str | None = None, max_output: int | None = None,
                 sampling: dict | None = None):
        # provider pins one openrouter provider with no fallbacks, so a sweep never mixes hosts
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=600, max_retries=0,
                             default_headers={"X-Title": "jev-context-filter-harness"} if provider else None)
        self.model = model
        body = {}
        if provider:
            body["provider"] = {"order": [provider], "only": [provider], "allow_fallbacks": False}
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        # sampling set in the request, the same in every configuration. top_k is not an openai field, so it rides the body.
        # so do min_p and chat_template_kwargs (vllm's), each only when set. presence_penalty is top level
        sampling = dict(sampling or {})
        for k in ("top_k", "min_p", "chat_template_kwargs"):
            if k in sampling:
                body[k] = sampling.pop(k)
        self.extra = {"extra_body": body} if body else {}
        # hosted means the request carries an extra body (a provider, reasoning effort, or a body sampling field)
        self.waits = HOSTED_WAITS if self.extra else LOCAL_WAITS
        # some hosts (novita) reserve a huge default output budget and reject long inputs unless this is set
        self.gen = {**({"max_tokens": max_output} if max_output else {}), **sampling}
        # a hosted copy of a local model gets the local context size, so overflows fail the same way
        self.max_context = max_context
        self.trace = trace
        self.completion_tokens = None  # the last call's

    def chat(self, messages: list[dict], tools: list[dict] | None = None, kind: str = "turn") -> tuple[dict, int]:
        """returns (assistant message as a dict, prompt tokens of this call). kind is turn or compaction"""
        t0 = time.monotonic()
        retries = 0
        while True:
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, **self.gen,
                                                        **({"tools": tools} if tools else {}), **self.extra)
                break
            except (APIStatusError, APIConnectionError) as e:
                if isinstance(e, APIStatusError) and too_long(e):
                    raise ContextOverflow(str(e)) from e
                if not retryable(e):
                    raise
                wait = self.waits[retries] if retries < len(self.waits) else None
                # one row per failed attempt, kind retry with the wait before the next, or error when none
                # is left. the call's own row then has retries. fixed waits: a Retry-After header is not read
                self.trace.llm({"kind": "retry" if wait is not None else "error", "call_kind": kind,
                                "attempt": retries + 1, "status": getattr(e, "status_code", None),
                                "error": f"{type(e).__name__}: {e}", "wait_s": wait,
                                "latency_ms": round((time.monotonic() - t0) * 1000)})
                if wait is None:
                    raise
                time.sleep(wait)
                retries += 1
        m = r.choices[0].message
        msg = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                 for c in m.tool_calls]
        u = r.usage
        self.completion_tokens = u.completion_tokens if u else None
        # a call past max_context is logged as rejected, outside the totals, as a local server bills nothing
        over = bool(self.max_context and u and u.prompt_tokens > self.max_context)
        # n_messages: the turn call saw the context's first n_messages
        # a compaction call logs its own messages
        self.trace.llm({"kind": "rejected" if over else kind, "latency_ms": round((time.monotonic() - t0) * 1000),
                        "model": r.model, "response_id": r.id,
                        **({"provider": r.provider} if getattr(r, "provider", None) else {}),
                        **({"retries": retries} if retries else {}),  # how many retries the call took
                        "n_messages": len(messages),
                        **({"messages": messages} if kind == "compaction" else {}),
                        "prompt_tokens": u.prompt_tokens if u else None,
                        "completion_tokens": u.completion_tokens if u else None,
                        "finish_reason": r.choices[0].finish_reason,
                        "reasoning": getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None),
                        "message": msg})
        if over:
            raise ContextOverflow(f"request ({u.prompt_tokens} tokens) exceeds the context size "
                                  f"({self.max_context} tokens)")
        return msg, (u.prompt_tokens if u else 0)
