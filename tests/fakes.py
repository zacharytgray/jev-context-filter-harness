"""the fake llm and the fake jev the tests share. nothing here calls a real model"""
import json
from pathlib import Path

import httpx

from s1harness.jev import Jev, Trace
from s1harness.levers import Levers, load_levers
from s1harness.loop import run

LEVERS_DIR = Path(__file__).parent.parent / "levers"
CONFIGS = ("plain", "relevance", "injection", "relevance-injection")


class FakeLLM:
    """replays scripted assistant messages. tokens: a number, or "chars" to count about four characters a token"""

    def __init__(self, replies, tokens=1000):
        self.replies = list(replies)
        self.seen = []
        self.kinds = []
        self.tokens = tokens
        self.completion_tokens = None

    def chat(self, messages, tools=None, kind="turn"):
        if kind == "turn":
            self.tools = tools
        self.seen.append([dict(m) for m in messages])
        self.kinds.append(kind)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        self.completion_tokens = len(json.dumps(r)) // 4 + 1
        if self.tokens == "chars":
            return r, (len(json.dumps(messages)) + len(json.dumps(tools or []))) // 4
        if callable(self.tokens):
            return r, self.tokens(messages)
        return r, self.tokens


class FakeJev(Jev):
    """the real Jev on a fake transport, so every request is logged and checked as in a live run. it answers by
    the state it is sent: relevance says yes to a chunk holding word, injection to a chunk holding plant, and the
    finder names the first and last line holding plant. fail names a filter whose requests get a 500, malformed one
    whose answers are broken, after the first fail_after requests of that filter are answered (every attempt counts,
    so a failing request is sent 4 times). finder="reversed" swaps the finder's first and last line, "block" names only the
    first run of plant lines, "miss" names the first line that holds no plant. a list of these is used one per finder request"""

    retry_waits = (0, 0, 0)  # the retries without the waits

    def __init__(self, trace, word="NEEDLE", plant="PLANT", fail=None, malformed=None, finder=None, fail_after=0):
        super().__init__("jev-test", trace, api_key="test-key")
        self.word, self.plant, self.fail, self.malformed, self.finder = word, plant, fail, malformed, finder
        self.fail_after = fail_after
        self.bodies = []
        self.http = httpx.Client(transport=httpx.MockTransport(self._handle))

    @staticmethod
    def kind(state: dict) -> str:
        return "finder" if "chunk" in state else "relevance" if "question" in state else "injection"

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        st, qs = body["state"], body["questions"]
        kind = self.kind(st)
        if self.fail == kind and len(self.sent(kind)) > self.fail_after:
            return httpx.Response(500, json={"error": "boom"})
        if kind == "relevance":
            answers = {q: {"noul": 0.9 if self.word in st["chunks"][int(q[1:])]["text"] else 0.1} for q in qs}
        elif kind == "injection":
            answers = {"q0": {"noul": 0.9 if self.plant in st["chunks"][0]["text"] else 0.1}}
        else:
            lines = st["chunk"]["lines"]
            mode = self.finder.pop(0) if isinstance(self.finder, list) else self.finder
            hits = [x["n"] for x in lines if self.plant in x["text"]] or [lines[0]["n"]]
            if mode == "block":  # stop at the first line after the first hit that holds no plant
                ns = [x["n"] for x in lines]
                k = ns.index(hits[0])
                while k + 1 < len(ns) and ns[k + 1] in hits:
                    k += 1
                hits = [hits[0], ns[k]]
            if mode == "miss":
                hits = [next((x["n"] for x in lines if self.plant not in x["text"]), lines[0]["n"])]
            first, last = (hits[-1], hits[0]) if mode == "reversed" else (hits[0], hits[-1])
            if mode == "reversed" and first == last:
                first, last = lines[-1]["n"], lines[0]["n"]
            answers = {"q0": {"probabilities": {str(x["n"]): 1.0 if x["n"] == first else 0.0 for x in lines}},
                       "q1": {"probabilities": {str(x["n"]): 1.0 if x["n"] == last else 0.0 for x in lines}}}
        if self.malformed == kind:
            answers = {q: {"noul": "high"} for q in qs}
        return httpx.Response(200, json={"answers": answers, "model": "jev-test",
                                         "usage": {"input_tokens": len(json.dumps(body)) // 4, "output_tokens": 1}})

    def sent(self, kind: str) -> list[dict]:
        return [b for b in self.bodies if self.kind(b["state"]) == kind]


def call(name, cid=None, **args):
    return {"id": cid or f"c_{name}_{len(args)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def reply(text):
    return {"role": "assistant", "content": text}


def calls(*cs):
    return {"role": "assistant", "content": "", "tool_calls": list(cs)}


def levers(name: str) -> Levers:
    return load_levers(LEVERS_DIR / f"{name}.toml")


def go(tmp_path, replies, config="plain", files=None, request="do the thing", tokens=1000, run_kw=None,
       **jev_kw):
    """one run in a fresh workdir. files maps a relative path to its text"""
    work = tmp_path / "work"
    work.mkdir()
    for name, text in (files or {}).items():
        (work / name).parent.mkdir(parents=True, exist_ok=True)
        (work / name).write_text(text)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trace = Trace(run_dir)
    llm = FakeLLM(replies, tokens)
    jev = FakeJev(trace, **jev_kw)
    lv = config if isinstance(config, Levers) else levers(config)
    res = run(request, work, lv, llm, jev, trace, **(run_kw or {}))
    return res, llm, jev, run_dir


def trace_rows(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines() if line.strip()]


def events(run_dir: Path, name: str) -> list[dict]:
    return [r for r in trace_rows(run_dir) if r.get("event") == name]


def tool_messages(run_dir: Path) -> list[str]:
    return [m["content"] for m in json.loads((run_dir / "transcript.json").read_text()) if m["role"] == "tool"]
