import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from .jev import Jev, Trace
from .levers import load_levers
from .llm import LLM
from .compaction import COMPACTION_BUDGET, CONTEXT_WINDOW
from .loop import run
from .pipeline import JEV_MODEL, MAX_TURNS

PLAIN = Path(__file__).resolve().parent.parent / "levers" / "plain.toml"
JEV_ERROR_EXIT = 3  # a jev error reached the loop (a safety net): the run is marked for rerun, not a crash


def main():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    ap = argparse.ArgumentParser(prog="s1h", description="run one task through the harness")
    ap.add_argument("request", help="the request, or @file to read it from a file")
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--levers", default=str(PLAIN), help="a configuration's toml file in levers/, default: plain loop")
    # for the two switches. the configuration's name follows the switches
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override one switch, repeatable, e.g. --set filters.relevance=true")
    ap.add_argument("--base-url", required=True,
                    help="the agent model's OpenAI-compatible endpoint, e.g. https://openrouter.ai/api/v1 (OpenRouter)")
    ap.add_argument("--model", required=True, help="the agent model's name at that endpoint, e.g. google/gemma-4-31b-it")
    ap.add_argument("--max-context", type=int, default=None,
                    help="fail a call whose prompt passes this many tokens, like a local server would")
    ap.add_argument("--context-window", type=int, default=CONTEXT_WINDOW,
                    help="the model's window in tokens. compaction triggers past the compaction budget, or the "
                         "window less max output and 4096 if that is smaller")
    # a run's own compaction budget, recorded in levers.json. the default is 32768
    ap.add_argument("--compaction-budget", type=int, default=COMPACTION_BUDGET,
                    help=f"compaction triggers past this many prompt tokens, default {COMPACTION_BUDGET}")
    ap.add_argument("--reasoning-effort", default=None, help="low, medium, or high, sent to the provider")
    ap.add_argument("--max-output", type=int, default=None, help="max_tokens per llm call, for hosts that need it")
    ap.add_argument("--provider", default=None, help="pin one openrouter provider, no fallbacks, e.g. deepinfra")
    ap.add_argument("--api-key-env", default=None, help="env var holding the llm api key, e.g. OPENROUTER_API_KEY")
    ap.add_argument("--temperature", type=float, default=None, help="sampling, sent as is. unset uses the host default")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    # for a vllm server. each is sent only when given
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--chat-template-kwargs", type=json.loads, default=None, metavar="JSON",
                    help='sent as chat_template_kwargs, e.g. \'{"enable_thinking": false}\'')
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    # a run's own turn limit, recorded in levers.json and result.json. the default is 40
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS, help=f"turns before the run ends, default {MAX_TURNS}")
    a = ap.parse_args()

    request = Path(a.request[1:]).read_text() if a.request.startswith("@") else a.request
    levers = load_levers(a.levers, a.set)
    run_dir = a.runs_dir / (time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 10**6:06d}")
    run_dir.mkdir(parents=True)
    trace = Trace(run_dir)
    llm = LLM(a.base_url, a.model, trace, os.environ[a.api_key_env] if a.api_key_env else "none", a.provider,
              a.max_context, a.reasoning_effort, a.max_output,
              {k: v for k, v in (("temperature", a.temperature), ("top_p", a.top_p), ("top_k", a.top_k),
                                 ("min_p", a.min_p), ("presence_penalty", a.presence_penalty),
                                 ("chat_template_kwargs", a.chat_template_kwargs)) if v is not None})
    jev = Jev(JEV_MODEL, trace)
    # the clients hold the keys now. drop them so the agent's bash never inherits them
    for k in ("TYPESAFE_API_KEY", a.api_key_env):
        if k:
            os.environ.pop(k, None)
    def on_term(*_):  # the runner's timeout. raise, so the loop writes the records
        raise TimeoutError("timeout: the run passed its time limit")
    signal.signal(signal.SIGTERM, on_term)
    res = run(request, a.workdir.resolve(), levers, llm, jev, trace, context_window=a.context_window,
              max_output=a.max_output, max_turns=a.max_turns, compaction_budget=a.compaction_budget)
    print(f"end: {res.end} after {res.turns} turns ({res.run_dir})\n\n{res.final_message}")
    if res.rerun:
        sys.exit(JEV_ERROR_EXIT)


if __name__ == "__main__":
    main()
