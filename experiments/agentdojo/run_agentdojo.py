"""AgentDojo episodes with Gemma 4 31B through OpenRouter (Novita, no fallbacks), under two defenses:
none, and jev (the injection filter, s1harness.pipeline.injection_filter, on every tool output).

run with the agentdojo venv and the repo root on PYTHONPATH, from the repo root (README.md next to this file):
  PYTHONPATH=$PWD experiments/agentdojo/.venv/bin/python experiments/agentdojo/run_agentdojo.py \
      --suite banking --user-tasks user_task_0 --injection-tasks injection_task_0 --attack important_instructions \
      --defense none,jev --out <dir> --max-usd-openrouter 0.05 --max-jev-tokens 20000 --note "..." [--dry-run]

files in --out: records.jsonl (one line per episode, the last line for a key wins), llm_calls.jsonl (every llm call),
jev_trace.jsonl (every jev request, chunk decision and jev error, each with the tool call id), defense.jsonl (one line
per tool output the filter saw, with the call id), runs.jsonl (one line per call of this script), agentdojo/ (agentdojo's own per-episode logs). the episode's lines in records, llm_calls,
jev_trace and defense carry `play`, one id per play of an episode, so a replay's lines never mix with the stopped
attempt's.

a failed jev request is sent again up to 3 times (1, 2, 4 s), then the filter falls back open: a failed
first-round request counts its chunk as clear, and a failed finder call or re-ask ends that chunk's loop with
the lines earlier rounds hid still hidden. each fallback is a jev_fallback event in jev_trace.jsonl, and the record
counts jev_failed_requests, jev_retries and jev_fallbacks. a JevError that still reaches the filter element is a
safety net: it stops the episode, whose record has the error, rerun true, and jev_error; the next call with the same
arguments plays it again, and the analysis leaves it out of every rate and lists it as a rerun.
--llm-config <json> points the runner at another model, a self-hosted vllm server included (see configure). without
it every setting is the gemma one above.
offline checks: test_run_agentdojo.py next to this file.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # the repo root, for s1harness
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import openai  # noqa: E402
import yaml  # noqa: E402
from agentdojo.agent_pipeline import (  # noqa: E402
    AgentPipeline,
    BasePipelineElement,
    InitQuery,
    SystemMessage,
    ToolsExecutionLoop,
    ToolsExecutor,
)
from agentdojo.agent_pipeline.agent_pipeline import load_system_message  # noqa: E402
from agentdojo.agent_pipeline.llms.openai_llm import _function_to_openai  # noqa: E402
from agentdojo.agent_pipeline.tool_execution import tool_result_to_str  # noqa: E402
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.functions_runtime import EmptyEnv, FunctionCall  # noqa: E402
from agentdojo.logging import LOGGER_STACK, Logger, OutputLogger, TraceLogger  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from agentdojo.types import (  # noqa: E402
    ChatAssistantMessage,
    ChatToolResultMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)
from pydantic import BaseModel  # noqa: E402

from s1harness import pipeline as P  # noqa: E402
from s1harness.jev import TIMEOUT as JEV_TIMEOUT  # noqa: E402
from s1harness.jev import Jev, JevError, est_tokens  # noqa: E402

VERSION = "v1.2.2"
BASE_URL = "https://openrouter.ai/api/v1"
KEY_ENV = "OPENROUTER_API_KEY"
JEV_KEY_ENV = "TYPESAFE_API_KEY"
MODEL = "google/gemma-4-31b-it"
PROVIDER = {"order": ["Novita"], "only": ["Novita"], "allow_fallbacks": False}
TEMPERATURE, TOP_P, TOP_K, MAX_TOKENS = 1.0, 0.95, 64, 8192
PRICE_IN, PRICE_OUT = 0.1355 / 1e6, 1.055 / 1e6  # openrouter dollars per token, gemma 4 31b on novita
JEV_PRICE_IN = 0.042 / 1e6  # jev dollars per input token, only for the dry run's estimate
JEV_MODEL = P.JEV_MODEL
# a hosted provider can be busy for minutes: wait it out like s1harness/llm.py, after the client's own retries
RATE_WAITS = [15, 30, 60, 60, 120, 120, 240]
CLIENT_MAX_RETRIES = 6
# the attack reads {model} from the pipeline name: "openai-compatible" gives "AI assistant". no "/" in names
PIPE_PREFIX = "openai-compatible_gemma-4-31b-it"
# vllm's sampling, sent only when --llm-config sets it
MIN_P = PRESENCE_PENALTY = CHAT_TEMPLATE_KWARGS = None
# openrouter's reasoning effort (low, high, ...), sent as reasoning.effort only when --llm-config sets it. the
# reasoning is logged in llm_calls.jsonl and never sent back, as the harness does
REASONING_EFFORT = None
# --llm-config's keys, each defaulting to the values above. a key left out keeps its default
LLM_DEFAULTS = {"base_url": BASE_URL, "api_key_env": KEY_ENV, "model": MODEL, "provider": "Novita",
                "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "max_tokens": MAX_TOKENS},
                "price_per_1m": {"input": 0.1355, "output": 1.055},  # PRICE_IN and PRICE_OUT per 1M
                "reasoning_effort": None}
# the attacks this runner plays (agentdojo 0.1.35), with the (user, model) names each must carry.
# none plays the user tasks alone
ATTACKS = {
    "none": None,
    "important_instructions": ("Emma Johnson", "AI assistant"),
}
SUITES = ("banking", "slack", "travel", "workspace")
# dry-run priors per episode, used until DIR holds 3 finished episodes of the same kind. prompt and jev from
# an early scouting estimate (chars/4, 5 turns: banking really takes about 3). completion is a guess, 200 per call
PRIORS = {"banking": {"prompt": 9000, "completion": 600, "jev_in": 1400},
          "slack": {"prompt": 7100, "completion": 1200, "jev_in": 1100},
          "travel": {"prompt": 22900, "completion": 1400, "jev_in": 1750},
          "workspace": {"prompt": 25000, "completion": 800, "jev_in": 3800}}
FINDER_PRIOR = 1500  # extra jev input per attacked episode for the line finder, a guess


# ---- defenses and episodes

@dataclass(frozen=True)
class Defense:
    kind: str  # none or jev

    @property
    def id(self) -> str:
        return "jev-injection" if self.kind == "jev" else self.kind

    @property
    def settings(self) -> dict:
        if self.kind == "jev":
            return {"filter": "injection", "wording": P.INJECTION_WORDING, "threshold": P.INJECTION_THRESHOLD,
                    "finder": P.FINDER, "finder_wording": P.FINDER_WORDING,
                    "max_calls_per_chunk": P.INJECTION_MAX_CALLS, "chunk_lines": P.CHUNK_LINES,
                    "chunk_chars": P.CHUNK_CHARS, "long_line_overlap": P.LONG_LINE_OVERLAP, "model": JEV_MODEL,
                    "name": "<tool name> output", "checks": "content and error text"}
        return {}


@dataclass(frozen=True)
class Episode:
    suite: str
    user_task: str
    injection_task: str | None
    attack: str
    defense: Defense

    @property
    def key(self) -> str:
        return "|".join([self.suite, self.user_task, self.injection_task or "none", self.attack, self.defense.id])

    @property
    def pipe_name(self) -> str:
        return f"{PIPE_PREFIX}_{self.defense.id}"


def record_key(rec: dict) -> str:
    return "|".join([rec["suite"], rec["user_task"], rec["injection_task"] or "none", rec["attack"],
                     rec["defense_id"]])


def _num(task_id: str) -> int:
    return int(task_id.rsplit("_", 1)[-1])


def pick_tasks(spec: str, available, prefix: str, what: str) -> list[str]:
    """'all', 'none', or a comma list of ids (a bare number means prefix + number)"""
    if spec == "all":
        return sorted(available, key=_num)
    if spec == "none":
        return []
    out = []
    for s in (x.strip() for x in spec.split(",")):
        if not s:
            continue
        s = f"{prefix}{s}" if s.isdigit() else s
        if s not in available:
            raise SystemExit(f"unknown {what} {s!r}")
        out.append(s)
    return out


def plan(suite, a) -> list[Episode]:
    """pairs in task order, each pair once per defense, so the defenses interleave"""
    uts = pick_tasks(a.user_tasks, suite.user_tasks, "user_task_", "user task")
    if a.injection_tasks is None and a.attack != "none":
        raise SystemExit("give --injection-tasks (a comma list or all) with an attack")
    its = pick_tasks(a.injection_tasks or "none", suite.injection_tasks, "injection_task_", "injection task")
    if a.attack == "none" and its:
        raise SystemExit("--attack none plays user tasks alone: pass --injection-tasks none")
    if a.attack != "none" and not its:
        raise SystemExit(f"--attack {a.attack} needs injection tasks")
    defenses = []
    for d in (x.strip() for x in a.defense.split(",")):
        if d in ("none", "jev"):
            defenses.append(Defense(d))
        elif d:
            raise SystemExit(f"unknown defense {d!r}, expected none or jev")
    if len(set(defenses)) != len(defenses):
        raise SystemExit("a defense is listed twice")
    pairs = [(u, i) for u in uts for i in (its or [None])]
    return [Episode(a.suite, u, i, a.attack, d) for u, i in pairs for d in defenses]


def check_names(name: str, attack) -> None:
    """the attack's user and model names are the ones agentdojo gives this pipeline"""
    want = ATTACKS[name]
    if want and (attack.user_name, attack.model_name) != want:
        raise SystemExit(f"{name}: attack names are {attack.user_name!r}, {attack.model_name!r}, expected {want}")


# ---- shared run state

class Writer:
    def __init__(self, out: Path):
        self.out, self.lock = out, threading.Lock()

    def append(self, name: str, rec: dict):
        line = json.dumps({"ts": round(time.time(), 3), **rec}, default=str)
        with self.lock, open(self.out / name, "a") as f:
            f.write(line + "\n")


class Totals:
    """running usage of this call, from the logged usage, for the caps"""

    def __init__(self):
        self.lock = threading.Lock()
        self.prompt = self.completion = self.jev_in = self.jev_out = 0

    def add_llm(self, p: int, c: int):
        with self.lock:
            self.prompt += p
            self.completion += c

    def add_jev(self, i: int, o: int):
        with self.lock:
            self.jev_in += i
            self.jev_out += o

    @property
    def usd(self) -> float:
        return self.prompt * PRICE_IN + self.completion * PRICE_OUT


@dataclass
class Ctx:
    a: argparse.Namespace
    out: Path
    suite: object
    writer: Writer
    totals: Totals
    commit: dict
    client: object = None  # openai client
    jev_http: object = None
    jev_key: str = ""
    injections: dict = field(default_factory=dict)


@dataclass
class EpisodeState:
    ep: Episode
    play: str = field(default_factory=lambda: uuid.uuid4().hex[:12])  # this play of the episode
    calls: int = 0
    attempts: int = 0
    prompt: int = 0
    completion: int = 0
    rate_limited: int = 0
    providers: Counter = field(default_factory=Counter)
    finish: Counter = field(default_factory=Counter)
    tool_calls: int = 0
    empty_args: int = 0
    bad_args: list = field(default_factory=list)
    pending_bad: dict = field(default_factory=dict)  # call id -> error text, for the executor of this turn
    seen_ids: set = field(default_factory=set)
    jev_in: int = 0
    jev_out: int = 0
    jev_requests: int = 0
    jev_failed: int = 0  # requests that failed every attempt
    jev_retries: int = 0
    jev_fallbacks: Counter = field(default_factory=Counter)  # by filter: injection, finder
    jev_usage_missing: int = 0
    jev_error: dict | None = None  # the error that stopped the episode
    provider_error: str | None = None  # the llm provider's failure that stopped the episode
    empty_replies: int = 0  # replies with no text and no tool call that the provider finished normally
    filtered: list = field(default_factory=list)


# ---- the llm element

def to_openai(m: dict) -> dict:
    """agentdojo message -> openai chat message with a plain string content. system goes as "system", not "developer" """
    role = m["role"]
    text = get_text_content_as_str(m["content"]) if m["content"] is not None else None
    if role in ("system", "user"):
        return {"role": role, "content": text or ""}
    if role == "assistant":
        out = {"role": "assistant", "content": text or ""}
        if m["tool_calls"]:
            out["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.function, "arguments": json.dumps(c.args)}}
                                 for c in m["tool_calls"]]
        return out
    if role == "tool":
        return {"role": "tool", "tool_call_id": m["tool_call_id"], "name": m["tool_call"].function,
                "content": m["error"] or text or ""}
    raise ValueError(f"unknown message role {role!r}")


def parse_args(raw) -> tuple[dict | None, str]:
    """(args, "") or (None, why). empty arguments mean a call with none"""
    if raw is None or not str(raw).strip():
        return {}, ""
    try:
        v = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON ({e.msg} at character {e.pos})"
    if not isinstance(v, dict):
        return None, f"JSON {type(v).__name__} instead of an object"
    return v, ""


def configure(cfg: dict) -> dict:
    """sets the llm's module settings from --llm-config's json over LLM_DEFAULTS, and returns the merged config. an
    empty config restores today's openrouter settings exactly. provider null pins none (a self-hosted server),
    api_key_env null sends no key"""
    global BASE_URL, KEY_ENV, MODEL, PROVIDER, TEMPERATURE, TOP_P, TOP_K, MAX_TOKENS, MIN_P, PRESENCE_PENALTY
    global CHAT_TEMPLATE_KWARGS, PRICE_IN, PRICE_OUT, PIPE_PREFIX, REASONING_EFFORT
    unknown = set(cfg) - set(LLM_DEFAULTS)
    if unknown:
        raise SystemExit(f"--llm-config: unknown keys {sorted(unknown)}")
    c = {**LLM_DEFAULTS, **cfg}
    s = {**LLM_DEFAULTS["sampling"], **(cfg.get("sampling") or {})}
    if cfg:  # a config sends top_p and top_k only when it sets them. left out or null: the server's own
        given = cfg.get("sampling") or {}
        s = {k: v for k, v in s.items() if k not in ("top_p", "top_k") or given.get(k) is not None}
    c["sampling"] = s
    BASE_URL, KEY_ENV, MODEL = c["base_url"], c["api_key_env"], c["model"]
    p = c["provider"]
    PROVIDER = {"order": [p], "only": [p], "allow_fallbacks": False} if p else None
    TEMPERATURE, TOP_P, TOP_K, MAX_TOKENS = s["temperature"], s.get("top_p"), s.get("top_k"), s["max_tokens"]
    MIN_P, PRESENCE_PENALTY = s.get("min_p"), s.get("presence_penalty")
    CHAT_TEMPLATE_KWARGS = s.get("chat_template_kwargs")
    REASONING_EFFORT = c["reasoning_effort"]
    PRICE_IN, PRICE_OUT = c["price_per_1m"]["input"] / 1e6, c["price_per_1m"]["output"] / 1e6
    # the model's own name after the last "/", so the default stays openai-compatible_gemma-4-31b-it
    PIPE_PREFIX = "openai-compatible_" + MODEL.split("/")[-1]
    return c


def request_params() -> dict:
    """the sampling and routing every llm call sends. the default is today's body, key for key"""
    extra = {**({"provider": PROVIDER} if PROVIDER else {}), **({"top_k": TOP_K} if TOP_K is not None else {}),
             **({"min_p": MIN_P} if MIN_P is not None else {}),
             **({"chat_template_kwargs": CHAT_TEMPLATE_KWARGS} if CHAT_TEMPLATE_KWARGS is not None else {}),
             **({"reasoning": {"effort": REASONING_EFFORT}} if REASONING_EFFORT else {})}
    return {"temperature": TEMPERATURE, **({"top_p": TOP_P} if TOP_P is not None else {}), "max_tokens": MAX_TOKENS,
            **({"presence_penalty": PRESENCE_PENALTY} if PRESENCE_PENALTY is not None else {}), "extra_body": extra}


def sampling() -> dict:
    """what a record keeps of request_params()"""
    return {"temperature": TEMPERATURE, **{k: v for k, v in (("top_p", TOP_P), ("top_k", TOP_K)) if v is not None},
            "max_tokens": MAX_TOKENS,
            **{k: v for k, v in (("min_p", MIN_P), ("presence_penalty", PRESENCE_PENALTY),
                                 ("chat_template_kwargs", CHAT_TEMPLATE_KWARGS),
                                 ("reasoning_effort", REASONING_EFFORT)) if v is not None}}


def fresh_id(call: int, j: int, seen: set) -> str:
    """an id for a tool call that came back with an empty or repeated one: 9 letters and digits, the only form
    mistral's tokenizer takes (gemma takes any). the same call and position give the same id"""
    k = 0
    while (cid := hashlib.sha256(f"{call}_{j}_{k}".encode()).hexdigest()[:9]) in seen:
        k += 1
    return cid


class OpenRouterLLM(BasePipelineElement):
    def __init__(self, ctx: Ctx, st: EpisodeState):
        self.ctx, self.st = ctx, st

    def _create(self, body: dict):
        waits = RATE_WAITS
        k = 0
        while True:
            try:
                return self.ctx.client.chat.completions.create(**body), k
            except openai.RateLimitError:
                if k >= len(waits):
                    raise
                time.sleep(waits[k])
                k += 1

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        st, ep = self.st, self.st.ep
        msgs = [to_openai(m) for m in messages]
        tools = [_function_to_openai(f) for f in runtime.functions.values()]
        body = {"model": MODEL, "messages": msgs, **request_params()}
        if tools:
            body |= {"tools": tools, "tool_choice": "auto"}
        st.calls += 1
        if len(messages) <= 2:  # system and user only: agentdojo started (or restarted) the episode
            st.attempts += 1
        base = {"episode": ep.key, "play": st.play, "defense": ep.defense.id, "call": st.calls, "attempt": st.attempts,
                "n_messages": len(msgs)}
        t0 = time.monotonic()
        try:
            r, k = self._create(body)
        except Exception as e:
            if isinstance(e, openai.APIError):  # http errors after the client's retries, timeouts, lost connections
                st.provider_error = f"{type(e).__name__}: {e}"[:500]
            self.ctx.writer.append("llm_calls.jsonl", {**base, "latency_ms": _ms(t0), "error": repr(e)})
            raise
        st.rate_limited += k
        u = r.usage
        p = (u.prompt_tokens or 0) if u else 0
        c = (u.completion_tokens or 0) if u else 0
        st.prompt += p
        st.completion += c
        self.ctx.totals.add_llm(p, c)
        provider = getattr(r, "provider", None)
        st.providers[str(provider)] += 1
        if not r.choices:  # openrouter can answer 200 with an error body and no choices
            st.provider_error = f"no choices: {getattr(r, 'error', None)}"[:500]
            self.ctx.writer.append("llm_calls.jsonl", {**base, "latency_ms": _ms(t0), "provider": provider,
                                                       "error": st.provider_error})
            raise RuntimeError(f"no choices in the response: {getattr(r, 'error', None)}")
        choice = r.choices[0]
        m = choice.message
        # a reply the provider did not finish (finish_reason error, or none at all) is its failure, not the model's
        # answer: agentdojo would score it as the episode's end
        if choice.finish_reason in (None, "error"):
            st.provider_error = (f"finish_reason {choice.finish_reason}: "
                                 f"{getattr(choice, 'error', None) or getattr(r, 'error', None)}")[:500]
            self.ctx.writer.append("llm_calls.jsonl", {**base, "latency_ms": _ms(t0), "provider": provider,
                                                       "response_id": r.id, "prompt_tokens": p,
                                                       "completion_tokens": c, "content": m.content,
                                                       "error": st.provider_error})
            raise RuntimeError(f"provider error: {st.provider_error}")
        if not (m.content or "").strip() and not m.tool_calls:
            st.empty_replies += 1
        st.finish[str(choice.finish_reason)] += 1
        calls, logged = [], []
        for j, tc in enumerate(m.tool_calls or []):
            name = (tc.function.name if tc.function else None) or ""
            raw = tc.function.arguments if tc.function else None
            args, why = parse_args(raw)
            cid = tc.id if tc.id and tc.id not in st.seen_ids else fresh_id(st.calls, j, st.seen_ids)
            st.seen_ids.add(cid)
            if raw is None or not str(raw).strip():
                st.empty_args += 1
            if why:
                # agentdojo would json.loads this and end the whole benchmark. here the call gets an error result
                st.pending_bad[cid] = f"Invalid arguments for {name}: {why}. The call was not run."
                st.bad_args.append({"call": st.calls, "id": cid, "name": name, "arguments": raw, "why": why})
            calls.append(FunctionCall(function=name, args=args if args is not None else {}, id=cid))
            logged.append({"id": cid, "name": name, "arguments": raw, **({"bad": why} if why else {})})
        st.tool_calls += len(calls)
        details = getattr(u, "completion_tokens_details", None) if u else None
        self.ctx.writer.append("llm_calls.jsonl", {
            **base, "latency_ms": _ms(t0), "model": r.model, "provider": provider, "response_id": r.id,
            "rate_limited": k, "prompt_tokens": p, "completion_tokens": c,
            "reasoning_tokens": getattr(details, "reasoning_tokens", None), "finish_reason": choice.finish_reason,
            "usd_est": round(p * PRICE_IN + c * PRICE_OUT, 7), "content": m.content,
            "reasoning": getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None),
            "tool_calls": logged})
        content = [text_content_block_from_string(m.content)] if m.content is not None else None
        out = ChatAssistantMessage(role="assistant", content=content, tool_calls=calls or None)
        return query, runtime, env, [*messages, out], extra_args


def _plain(x):
    """models to dicts and tuples to lists, all the way down, so yaml.safe_dump takes it"""
    if isinstance(x, BaseModel):
        return x.model_dump()
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def yaml_tool_output(result) -> str:
    """--tool-format yaml. agentdojo's tool_result_to_str yaml-dumps a model or a list of models, strs and ints, but
    str()s a dict, so a value's newlines come out as literal \\n on one line (travel's reviews). here a dict, and a
    list agentdojo would refuse (one holding a dict, say), go through the same yaml.safe_dump. the rest is agentdojo's
    own formatting"""
    if isinstance(result, dict) or (isinstance(result, list) and not all(
            type(v) in (str, int) or isinstance(v, BaseModel) for v in result)):
        return yaml.safe_dump(_plain(result)).strip()
    return tool_result_to_str(result)


TOOL_FORMATS = {"str": tool_result_to_str, "yaml": yaml_tool_output}  # str: agentdojo's default


class ArgCheckExecutor(ToolsExecutor):
    """agentdojo's executor, except a call whose arguments did not parse gets an error result and is not run. the
    formatter turns a tool's result into the text the agent gets"""

    def __init__(self, st: EpisodeState, formatter=tool_result_to_str):
        super().__init__(tool_output_formatter=formatter)
        self.st = st

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        bad, self.st.pending_bad = self.st.pending_bad, {}
        if not bad or not messages or messages[-1]["role"] != "assistant" or not messages[-1]["tool_calls"]:
            return super().query(query, runtime, env, messages, extra_args)
        last, results = messages[-1], []
        for tc in last["tool_calls"]:
            if tc.id in bad:
                results.append(ChatToolResultMessage(role="tool", content=[text_content_block_from_string("")],
                                                     tool_call_id=tc.id, tool_call=tc, error=bad[tc.id]))
            else:  # the good calls run one at a time through agentdojo's own code, in order
                one = {**last, "tool_calls": [tc]}
                _, _, env, got, _ = super().query(query, runtime, env, [*messages[:-1], one], extra_args)
                results.append(got[-1])
        return query, runtime, env, [*messages, *results], extra_args


# ---- the jev defense: the injection filter

class EpisodeTrace:
    """what Jev writes to, per episode: every request goes to jev_trace.jsonl with its call id, and the usage adds up
    for the record and the cap. the filter's chunk rows and a jev error go there too"""

    def __init__(self, ctx: Ctx, st: EpisodeState):
        self.ctx, self.st, self.lock = ctx, st, threading.Lock()  # the filter asks jev from up to 6 threads

    def request(self, rec: dict):
        usage = rec.get("usage") or {}
        i = usage.get("input_tokens")
        missing = i is None
        if missing:  # a failed request has no usage. count the estimate, so the cap still moves
            i = est_tokens(rec.get("body", {})) * (rec.get("attempts") or 1)  # every attempt was posted
        o = usage.get("output_tokens") or 0
        with self.lock:
            self.st.jev_usage_missing += missing
            self.st.jev_in += i
            self.st.jev_out += o
            self.st.jev_requests += 1
            self.st.jev_failed += "error" in rec
            self.st.jev_retries += (rec.get("attempts") or 1) - 1
        self.ctx.totals.add_jev(i, o)
        self.ctx.writer.append("jev_trace.jsonl", {"episode": self.st.ep.key, "play": self.st.play, "kind": "request",
                                                   **rec})

    def chunk(self, call_id: str, rec: dict):
        self.ctx.writer.append("jev_trace.jsonl", {"episode": self.st.ep.key, "play": self.st.play, "kind": "chunk",
                                                   "call_id": call_id, **rec})

    def event(self, **rec):
        if rec.get("event") == "jev_fallback":
            with self.lock:
                self.st.jev_fallbacks[rec.get("filter")] += 1
        self.ctx.writer.append("jev_trace.jsonl", {"episode": self.st.ep.key, "play": self.st.play, "kind": "event",
                                                   **rec})


class EpisodeJev(Jev):
    """the harness's Jev client, with the run's shared http client and the episode's trace"""

    def __init__(self, trace: EpisodeTrace, http, key: str):
        super().__init__(JEV_MODEL, trace, key)
        self.http.close()
        self.http = http


class InjectionFilter(BasePipelineElement):
    """after the executor: every new tool output goes through the injection filter, its content and its error
    text (the error is what the agent gets when there is one). the request is agentdojo's `query`, the user task's
    prompt. a failed jev request falls back inside the filter. a JevError that still gets here is a safety net: it is
    logged and raised, and the episode stops"""

    def __init__(self, ctx: Ctx, st: EpisodeState):
        self.ctx, self.st = ctx, st
        self.trace = EpisodeTrace(ctx, st)
        self.jev = EpisodeJev(self.trace, ctx.jev_http, ctx.jev_key)

    def _check(self, query: str, text: str, name: str, cid: str, part: str) -> tuple[str, list[dict]]:
        try:
            new, records = P.injection_filter(self.jev, query, text, name, call_id=cid)
        except JevError as e:
            self.st.jev_error = {"call_id": cid, "filter": e.filter, "error": str(e), "part": part}
            self.trace.event(event="jev_error", **self.st.jev_error)
            raise
        for r in records:
            self.trace.chunk(cid, {**r, "part": part})
        return new, records

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        n = 0
        for m in reversed(messages):
            if m["role"] != "tool":
                break
            n += 1
        if not n:
            return query, runtime, env, messages, extra_args
        out = list(messages[:-n])
        for m in messages[-n:]:
            cid, tool = m["tool_call_id"], m["tool_call"].function
            name = f"{tool} output"
            rq0, in0 = self.st.jev_requests, self.st.jev_in
            content = get_text_content_as_str(m["content"] or [])
            error = m["error"] or ""
            new_content, records = self._check(query, content, name, cid, "content")
            new = dict(m)
            if new_content != content:
                new["content"] = [text_content_block_from_string(new_content)]
            if error:
                new_error, more = self._check(query, error, name, cid, "error")
                records = records + more
                new["error"] = new_error
            flagged = [r for r in records if r["decision"] == "flagged"]
            before, after = error or content, new.get("error") or new_content  # the agent gets the error if any
            rec = {"call_id": cid, "tool": tool, "shown": "error" if error else "content", "chars": len(before),
                   "chunks": len(records), "flagged": len(flagged),
                   "lines_withheld": sum(b - a + 1 for r in flagged for a, b in r["hidden"]),
                   "finder_spans": [r.get("span") for r in flagged],
                   "hidden_whole": sum(hid_whole(r) for r in flagged),
                   # the injection loop: questions asked and chunks that hit the call ceiling
                   "injection_rounds": sum(len(r["rounds"]) for r in records),
                   "stuck": sum(r.get("outcome") == "stuck" for r in flagged),
                   # chunks where a request fell back: counted clear, or the loop stopped early
                   "fallbacks": sum(bool(r.get("fallback")) for r in records),
                   "jev_requests": self.st.jev_requests - rq0, "jev_input_tokens": self.st.jev_in - in0,
                   "changed": new_content != content or new.get("error") != m["error"]}
            self.st.filtered.append(rec)
            self.ctx.writer.append("defense.jsonl", {"episode": self.st.ep.key, "play": self.st.play,
                                                     "defense": self.st.ep.defense.id, "call": self.st.calls, **rec,
                                                     "before": before,
                                                     **({"after": after} if after != before else {})})
            out.append(new)
        return query, runtime, env, out, extra_args



def hid_whole(r: dict) -> bool:
    """a flagged chunk whose final outcome hid all its lines: no span in any round, the call ceiling, or a long line.
    the first round's span alone misses a chunk the loop hid whole in a later round"""
    return r.get("outcome") in ("no span", "stuck", "long line")


# ---- one episode

def build_pipeline(ctx: Ctx, st: EpisodeState) -> AgentPipeline:
    llm = OpenRouterLLM(ctx, st)
    kind = st.ep.defense.kind
    defense = InjectionFilter(ctx, st) if kind == "jev" else None
    loop = [ArgCheckExecutor(st, TOOL_FORMATS[ctx.a.tool_format])] + ([defense] if defense else []) + [llm]
    pipe = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), llm, ToolsExecutionLoop(loop)])
    pipe.name = st.ep.pipe_name
    return pipe


def end_state(messages) -> str:
    if not messages or messages[-1]["role"] != "assistant":
        return "no_answer"
    last = messages[-1]
    if last["tool_calls"]:
        return "max_iters"  # agentdojo's loop stops after 15 rounds
    return "no_answer" if last["content"] is None else "answered"


def run_episode(ctx: Ctx, ep: Episode) -> dict:
    """plays one episode and returns its record. never raises"""
    LOGGER_STACK.set([])  # the default stack list is shared by every thread
    st = EpisodeState(ep)
    suite = ctx.suite
    t0 = time.monotonic()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    utility = security = None
    error = tb = None
    messages = []
    try:
        ut = suite.get_user_task_by_id(ep.user_task)
        it = suite.get_injection_task_by_id(ep.injection_task) if ep.injection_task else None
        injections = ctx.injections.get((ep.user_task, ep.injection_task), {})
        pipe = build_pipeline(ctx, st)
        with OutputLogger(str(ctx.out / "agentdojo")):  # agentdojo's TraceLogger needs an entered logger
            with TraceLogger(delegate=Logger.get(), suite_name=suite.name, user_task_id=ut.ID,
                             injection_task_id=it.ID if it else None, injections=injections, attack_type=ep.attack,
                             pipeline_name=pipe.name, benchmark_version=VERSION) as logger:
                try:
                    utility, security = suite.run_task_with_pipeline(pipe, ut, it, injections)
                except Exception as e:  # a jev error lands here too: the episode stops
                    error, tb = f"{type(e).__name__}: {e}", traceback.format_exc(limit=8)
                    logger.log_error(error)
                logger.set_contextarg("utility", utility)
                logger.set_contextarg("security", security)
                messages = list(logger.messages)
    except Exception as e:  # anything outside the episode itself: still one record, and the run goes on
        error, tb = error or f"{type(e).__name__}: {e}", tb or traceback.format_exc(limit=8)
    g = st.filtered
    rec = {
        "suite": ep.suite, "user_task": ep.user_task, "injection_task": ep.injection_task, "attack": ep.attack,
        "play": st.play, "defense": ep.defense.kind, "defense_id": ep.defense.id, "defense_settings": ep.defense.settings,
        "benchmark_version": VERSION, "model": MODEL, "provider_pin": PROVIDER["only"][0] if PROVIDER else None,
        "sampling": sampling(),
        "utility": None if error else utility,
        # agentdojo's meaning: True means the injection task succeeded. None with no injection task or an error
        "security": None if error or ep.injection_task is None else security,
        "end_state": ("jev error" if st.jev_error else "provider error" if error and st.provider_error
                      else "error" if error else end_state(messages)), "error": error,
        # the provider failed (an http error after retries, no choices, finish_reason error or none): never a result
        "provider_error": st.provider_error if error else None, "empty_replies": st.empty_replies,
        # a jev error is not a result: the episode is played again and replaces this record
        **({"rerun": True, "jev_error": st.jev_error} if st.jev_error else {}),
        "turns": st.calls, "attempts": st.attempts, "tool_calls": st.tool_calls,
        "bad_tool_args": st.bad_args, "empty_tool_args": st.empty_args,
        "llm_prompt_tokens": st.prompt, "llm_completion_tokens": st.completion,
        "llm_usd_est": round(st.prompt * PRICE_IN + st.completion * PRICE_OUT, 7),
        "providers": dict(st.providers), "finish_reasons": dict(st.finish), "rate_limited": st.rate_limited,
        "jev_input_tokens": st.jev_in, "jev_output_tokens": st.jev_out, "jev_requests": st.jev_requests,
        "jev_failed_requests": st.jev_failed, "jev_retries": st.jev_retries,
        "jev_fallbacks": {f: st.jev_fallbacks[f] for f in ("injection", "finder")},
        "jev_usage_missing": st.jev_usage_missing,
        "injection_filter": None if ep.defense.kind != "jev" else {
            "outputs": len(g), "chunks": sum(x["chunks"] for x in g), "flagged": sum(x["flagged"] for x in g),
            "outputs_flagged": sum(x["flagged"] > 0 for x in g), "lines_withheld": sum(x["lines_withheld"] for x in g),
            "hidden_whole": sum(x["hidden_whole"] for x in g), "injection_rounds": sum(x["injection_rounds"] for x in g),
            "stuck": sum(x["stuck"] for x in g), "chunks_fallback": sum(x["fallbacks"] for x in g),
            "errors_checked": sum(x["shown"] == "error" for x in g),
            "outputs_changed": sum(x["changed"] for x in g)},
        "injection_vectors": sorted(ctx.injections.get((ep.user_task, ep.injection_task), {})),
        "wall_s": round(time.monotonic() - t0, 2), "started": started, "commit": ctx.commit, "note": ctx.a.note,
        "tool_format": ctx.a.tool_format,
        "log": str(Path("agentdojo") / ep.pipe_name / ep.suite / ep.user_task / ep.attack
                   / f"{ep.injection_task or 'none'}.json"),
    }
    if tb:
        rec["traceback"] = tb
    return rec


# ---- setup

def _ms(t0: float) -> int:
    return round((time.monotonic() - t0) * 1000)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def commit_info() -> dict:
    def git(*args, cwd=ROOT):
        try:
            return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "runner_sha": _sha(Path(__file__)), "pipeline_sha": _sha(ROOT / "s1harness" / "pipeline.py"),
            "jev_sha": _sha(ROOT / "s1harness" / "jev.py"),
            "agentdojo_commit": git("rev-parse", "HEAD", cwd=HERE / "vendor" / "agentdojo") or "089ed468"}


def load_env(env_file: str | None):
    """the worktree's .env, then the main checkout's. variables already set win"""
    from dotenv import load_dotenv
    if env_file == "none":
        return
    files = [Path(env_file)] if env_file else [ROOT / ".env"]
    if not env_file:
        try:
            common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=10).stdout.strip()
            if common:
                files.append(Path(common).parent / ".env")
        except Exception:
            pass
    for f in files:
        if f.is_file():
            load_dotenv(f, override=False)


def read_records(out: Path) -> list[dict]:
    p = out / "records.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def estimate(suite: str, ep: Episode, done: list[dict]) -> dict:
    """per-episode usage: the mean of finished episodes of the same kind in DIR when there are 3, else the priors"""
    same = [r for r in done if not r["error"] and r["suite"] == suite and r["defense_id"] == ep.defense.id
            and (r["injection_task"] is None) == (ep.injection_task is None)]
    if len(same) >= 3:
        mean = lambda k: sum(r[k] for r in same) / len(same)  # noqa: E731
        return {"prompt": mean("llm_prompt_tokens"), "completion": mean("llm_completion_tokens"),
                "jev_in": mean("jev_input_tokens"), "source": f"mean of {len(same)} in DIR"}
    p = PRIORS[suite]
    jev = 0
    if ep.defense.kind == "jev":
        jev = p["jev_in"] + (FINDER_PRIOR if ep.injection_task else 0)
    return {"prompt": p["prompt"], "completion": p["completion"], "jev_in": jev, "source": "prior"}


def dry_run(a, eps: list[Episode], done_keys: set, done: list[dict]) -> dict:
    todo = [e for e in eps if e.key not in done_keys]
    print(f"{len(eps)} episodes planned, {len(eps) - len(todo)} already in {a.out}/records.jsonl, "
          f"{len(todo)} to play. suite {a.suite}, attack {a.attack}, benchmark {VERSION}")
    for e in eps:
        print(f"  {'done ' if e.key in done_keys else 'play '} {e.key}")
    est = [estimate(a.suite, e, done) for e in todo]
    p = sum(x["prompt"] for x in est)
    c = sum(x["completion"] for x in est)
    j = sum(x["jev_in"] for x in est)
    usd = p * PRICE_IN + c * PRICE_OUT
    by = Counter(e.defense.id for e in todo)
    print("by defense: " + ", ".join(f"{k} {v}" for k, v in by.items()))
    print(f"estimate: {p:,.0f} prompt + {c:,.0f} completion tokens, OpenRouter ${usd:.4f} "
          f"(${PRICE_IN * 1e6:.4f}/M in, ${PRICE_OUT * 1e6:.3f}/M out); Jev {j:,.0f} input tokens "
          f"(about ${j * JEV_PRICE_IN:.4f} at ${JEV_PRICE_IN * 1e6:.3f}/M). sources: "
          + ", ".join(f"{k} {v}" for k, v in Counter(x["source"] for x in est).items()))
    print("agentdojo replays an episode whose final answer is empty up to 3 times, so the worst case is 3x.")
    print(f"tool format: {a.tool_format}")
    print(f"caps: OpenRouter ${a.max_usd_openrouter}, Jev {a.max_jev_tokens} input tokens; up to {a.workers} "
          "episodes in flight when a cap is reached still finish")
    return {"episodes": len(eps), "to_play": len(todo), "prompt": p, "completion": c, "usd": usd, "jev_in": j}


def summary(recs: list[dict]) -> str:
    lines = []
    for d in sorted({r["defense_id"] for r in recs}):
        rs = [r for r in recs if r["defense_id"] == d]
        ok = [r for r in rs if not r["error"]]
        att = [r for r in ok if r["injection_task"]]
        u = sum(bool(r["utility"]) for r in ok)
        s = sum(bool(r["security"]) for r in att)
        rerun = sum(bool(r.get("rerun")) for r in rs)
        falls = sum(sum((r.get("jev_fallbacks") or {}).values()) for r in rs)
        lines.append(f"  {d}: {len(rs)} episodes, {len(rs) - len(ok)} errors"
                     + (f" ({rerun} jev errors, to play again)" if rerun else "")
                     + (f", {falls} jev fallbacks" if falls else "") + f", utility {u}/{len(ok)}"
                     + (f", attack succeeded {s}/{len(att)}" if att else "")
                     + f", prompt {sum(r['llm_prompt_tokens'] for r in rs):,}, completion "
                       f"{sum(r['llm_completion_tokens'] for r in rs):,}, jev in {sum(r['jev_input_tokens'] for r in rs):,}")
    return "\n".join(lines)


def build_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suite", required=True, choices=SUITES)
    ap.add_argument("--user-tasks", required=True, help="comma list (user_task_3 or 3) or all")
    ap.add_argument("--injection-tasks", default=None, help="comma list, all, or none (the default with --attack none)")
    ap.add_argument("--attack", required=True, choices=tuple(ATTACKS))
    ap.add_argument("--defense", required=True,
                    help="comma list of none and jev (the injection filter, fixed). episodes interleave them")
    ap.add_argument("--workers", type=int, default=4, help="episodes at once")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-usd-openrouter", type=float, default=None,
                    help="stop launching episodes once this call's OpenRouter dollars reach it (required to play). "
                         "episodes already running finish, so the total can pass it by up to --workers episodes")
    ap.add_argument("--max-jev-tokens", type=int, default=None,
                    help="stop launching episodes once this call's Jev input tokens reach it (required with jev). "
                         "can be passed the same way")
    ap.add_argument("--max-consecutive-errors", type=int, default=5,
                    help="stop launching after this many errors in a row under any one defense, jev errors included")
    ap.add_argument("--retry-errors", action="store_true",
                    help="play again episodes whose last record is an error. jev errors are always played again")
    ap.add_argument("--env-file", default=None, help="a .env to load, or none. default: worktree, then main checkout")
    ap.add_argument("--dry-run", action="store_true", help="list the episodes and an estimated cost, spend nothing")
    ap.add_argument("--note", required=True, help="why this call runs, kept in every record")
    ap.add_argument("--llm-config", type=Path, default=None,
                    help="a json file over LLM_DEFAULTS: base_url, api_key_env (null: no key), model, provider (null: "
                         "no pin), sampling (vllm's min_p, presence_penalty, chat_template_kwargs too; top_p and top_k "
                         "sent only when it sets them), price_per_1m. default: gemma 4 31b on openrouter, novita")
    ap.add_argument("--tool-format", default="str", choices=tuple(TOOL_FORMATS),
                    help="how a tool's result becomes text. str (default): agentdojo's own, which str()s a dict onto "
                         "one line. yaml: dicts and lists of dicts yaml-dumped like agentdojo's models. kept in every "
                         "record")
    a = ap.parse_args(argv)
    if a.workers < 1:
        ap.error("--workers must be at least 1")
    return a


def done_keys_of(done: list[dict], retry_errors: bool) -> set:
    """keys whose last record stands. a jev error that stopped an episode (the safety net) never stands: it is played
    again"""
    last = {record_key(r): r for r in done}
    return {k for k, r in last.items() if not r.get("rerun") and not (retry_errors and r["error"])}


def main(argv=None, openai_http=None, jev_http=None) -> list[dict]:
    """returns the records written by this call. openai_http and jev_http are httpx clients for offline tests"""
    a = build_args(argv)
    llm_cfg = configure(json.loads(a.llm_config.read_text()) if a.llm_config else {})
    suite = get_suite(VERSION, a.suite)
    eps = plan(suite, a)
    a.out.mkdir(parents=True, exist_ok=True)
    done = read_records(a.out)
    formats = {r.get("tool_format", "str") for r in done}  # records before the flag are str
    if formats - {a.tool_format}:
        raise SystemExit(f"{a.out} holds records with tool format {sorted(formats)}, not {a.tool_format}: "
                         "use another --out")
    # an episode's key has no model in it, so one --out holds one model's records
    if {r.get("model", MODEL) for r in done} - {MODEL}:
        raise SystemExit(f"{a.out} holds records of {sorted({r.get('model') for r in done})}, not {MODEL}: "
                         "use another --out")
    done_keys = done_keys_of(done, a.retry_errors)
    if a.dry_run:
        dry_run(a, eps, done_keys, done)
        return []
    kinds = {e.defense.kind for e in eps}
    if a.max_usd_openrouter is None:
        raise SystemExit("--max-usd-openrouter is required to play episodes")
    if "jev" in kinds and a.max_jev_tokens is None:
        raise SystemExit("--max-jev-tokens is required with the jev defense")
    todo = [e for e in eps if e.key not in done_keys]
    load_env(a.env_file)
    if KEY_ENV and not os.environ.get(KEY_ENV):
        raise SystemExit(f"{KEY_ENV} is not set")
    if "jev" in kinds and not os.environ.get(JEV_KEY_ENV):
        raise SystemExit(f"{JEV_KEY_ENV} is not set")
    ctx = Ctx(a, a.out, suite, Writer(a.out), Totals(), commit_info())
    ctx.client = openai.OpenAI(base_url=BASE_URL, api_key=os.environ[KEY_ENV] if KEY_ENV else "none", timeout=600,
                               max_retries=CLIENT_MAX_RETRIES,
                               default_headers={"X-Title": "jev-context-filter-harness"} if PROVIDER else None,
                               http_client=openai_http)
    ctx.jev_http = jev_http or httpx.Client(timeout=JEV_TIMEOUT)
    ctx.jev_key = os.environ.get(JEV_KEY_ENV, "")
    if a.attack != "none":
        ref = AgentPipeline([])
        ref.name = PIPE_PREFIX
        attack = load_attack(a.attack, suite, ref)
        check_names(a.attack, attack)
        for e in todo:
            if (e.user_task, e.injection_task) not in ctx.injections:
                ctx.injections[(e.user_task, e.injection_task)] = attack.attack(
                    suite.get_user_task_by_id(e.user_task), suite.get_injection_task_by_id(e.injection_task))
    run = {"kind": "start", "argv": sys.argv[1:] if argv is None else list(argv), "note": a.note,
           "tool_format": a.tool_format, **({"llm": llm_cfg} if a.llm_config else {}),
           "planned": len(eps), "to_play": len(todo), "commit": ctx.commit, "prior_records": len(done)}
    ctx.writer.append("runs.jsonl", run)
    print(f"{len(todo)} of {len(eps)} episodes to play into {a.out} ({len(eps) - len(todo)} already recorded)",
          flush=True)

    recs, stop, errors_in_row = [], "", Counter()  # per defense: interleaved defenses would reset each other
    pending = list(todo)
    running = {}
    with ThreadPoolExecutor(a.workers) as ex:
        while pending or running:
            while pending and len(running) < a.workers and not stop:
                if ctx.totals.usd >= a.max_usd_openrouter:
                    stop = f"OpenRouter cap reached: ${ctx.totals.usd:.4f} of ${a.max_usd_openrouter}"
                elif a.max_jev_tokens is not None and ctx.totals.jev_in >= a.max_jev_tokens:
                    stop = f"Jev cap reached: {ctx.totals.jev_in} of {a.max_jev_tokens} input tokens"
                elif max(errors_in_row.values(), default=0) >= a.max_consecutive_errors:
                    d, n = errors_in_row.most_common(1)[0]
                    stop = f"{n} episodes in a row under {d} ended in an error"
                else:
                    e = pending.pop(0)
                    running[ex.submit(run_episode, ctx, e)] = e
            if stop and pending:
                print(f"stopping launches: {stop}. {len(pending)} not played, {len(running)} still running",
                      flush=True)
                pending = []
            if not running:
                break
            try:
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
            except KeyboardInterrupt:  # a second ctrl-c ends the process and loses the running records
                stop = stop or "interrupted"
                print(f"interrupted: waiting for {len(running)} running episodes, so their records are kept",
                      flush=True)
                pending = []
                continue
            for f in finished:
                e = running.pop(f)
                rec = f.result()
                ctx.writer.append("records.jsonl", rec)
                recs.append(rec)
                errors_in_row[e.defense.id] = errors_in_row[e.defense.id] + 1 if rec["error"] else 0
                print(f"[{len(recs)}/{len(todo)}] {e.key}: "
                      + (f"JEV ERROR, to play again: {rec['error'][:200]}" if rec.get("rerun") else
                         f"ERROR {rec['error'][:200]}" if rec["error"] else
                         f"utility {rec['utility']} security {rec['security']} {rec['end_state']}")
                      + f", {rec['turns']} calls, ${rec['llm_usd_est']:.5f}, jev in {rec['jev_input_tokens']}, "
                        f"{rec['wall_s']}s. total ${ctx.totals.usd:.4f}, jev in {ctx.totals.jev_in}", flush=True)
    ctx.writer.append("runs.jsonl", {"kind": "end", "note": a.note, "tool_format": a.tool_format, "played": len(recs),
                                     "stopped": stop or None,
                                     "prompt": ctx.totals.prompt, "completion": ctx.totals.completion,
                                     "usd_est": round(ctx.totals.usd, 6), "jev_in": ctx.totals.jev_in,
                                     "jev_out": ctx.totals.jev_out,
                                     "rerun": sum(bool(r.get("rerun")) for r in recs)})
    print(f"done: {len(recs)} played{f', stopped: {stop}' if stop else ''}. OpenRouter about ${ctx.totals.usd:.4f} "
          f"({ctx.totals.prompt:,} prompt, {ctx.totals.completion:,} completion), Jev {ctx.totals.jev_in:,} input tokens")
    if recs:
        print(summary(recs))
    return recs


if __name__ == "__main__":
    main()
