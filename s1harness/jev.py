import json
import os
import threading
import time
from pathlib import Path

import httpx

URL = "https://api.typesafe.ai/v1/systemone"
TIMEOUT = 10.0
# a failed request is sent again 3 times, after 1, 2 and 4 s, before the filter falls back
RETRY_WAITS = (1, 2, 4)


def est_tokens(obj) -> int:
    text = obj if isinstance(obj, str) else json.dumps(obj)
    return len(text) // 4 + 1


class JevError(Exception):
    """a jev request that failed every attempt: an http error, a timeout, bad json, or a missing or malformed answer.
    the filters catch it and fall back open"""

    def __init__(self, message: str, call_id: str | None = None, filter: str | None = None, attempts: int = 1,
                 chunks: list[int] | None = None):
        super().__init__(message)
        self.call_id = call_id
        self.filter = filter
        self.attempts = attempts
        self.chunks = chunks


class Trace:
    """one json line per jev request, chunk, or event, plus llm calls in their own file"""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.turn = 0
        self._jev = open(run_dir / "trace.jsonl", "a")
        self._llm = open(run_dir / "llm.jsonl", "a")
        self._lock = threading.Lock()  # the injection filter asks jev from several threads

    def _write(self, f, rec: dict):
        with self._lock:
            f.write(json.dumps({"ts": time.time(), "turn": self.turn, **rec}) + "\n")
            f.flush()

    def request(self, rec: dict):
        self._write(self._jev, {"kind": "request", **rec})

    def chunk(self, call_id: str | None, rec: dict):
        self._write(self._jev, {"kind": "chunk", "call_id": call_id, **rec})

    def event(self, **rec):
        self._write(self._jev, {"kind": "event", **rec})

    def llm(self, rec: dict):
        self._write(self._llm, rec)


class Jev:
    retry_waits = RETRY_WAITS

    def __init__(self, model: str, trace: Trace, api_key: str | None = None):
        self.model = model
        self.trace = trace
        self.key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.http = httpx.Client(timeout=TIMEOUT)
        self._lock = threading.Lock()
        self.requests = 0  # run-wide, so request ids follow send order
        self.input_tokens = 0

    def _attempt(self, body: dict, questions: dict) -> tuple[dict, dict | None, dict, list[str]]:
        """one post. returns (the reply, its usage, its answers, the questions with no well-formed answer), or raises
        httpx.HTTPError or ValueError"""
        r = self.http.post(URL, json=body, headers={"Authorization": f"Bearer {self.key}"})
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise ValueError("response is not an object")
        answers = data.get("answers") if isinstance(data.get("answers"), dict) else {}
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        # every question needs a well-formed answer. no partial answers reach a filter
        bad = [q for q in questions if not (q in answers and _well_formed(answers[q], questions[q]))]
        return data, usage, answers, bad

    def ask(self, state, questions: dict, filter: str, call_id: str | None = None,
            chunks: list[int] | None = None) -> dict:
        """one request, tried up to 4 times. returns every answer keyed by question id, or raises JevError once the
        last attempt fails. filter is relevance, injection, or finder; chunks are the chunk ids it covers, one per
        question. one trace row per request, with its attempts and every failed attempt's error"""
        body = {"model": self.model, "state": state, "questions": questions}
        with self._lock:
            self.requests += 1
            rid = self.requests
        rec = {"call_id": call_id, "request_id": rid, "filter": filter, "chunks": chunks, "t_start": time.time(),
               "body": body}
        t0 = time.monotonic()
        errors, spent, out = [], {}, {}
        for attempt, wait in enumerate((*self.retry_waits, None), 1):
            try:
                data, usage, answers, bad = self._attempt(body, questions)
            except (httpx.HTTPError, ValueError) as e:
                errors.append(repr(e))
                out = {}
            else:
                # the row's usage sums every attempt that reported one, since each is billed
                for k, v in (usage or {}).items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        spent[k] = spent.get(k, 0) + v
                with self._lock:
                    self.input_tokens += (usage or {}).get("input_tokens") or 0
                out = {"model": data.get("model"), "answers": answers}
                if not bad:
                    break
                errors.append(f"malformed answers: {', '.join(bad)}")
                out["malformed"] = bad
            if wait is None:
                break
            time.sleep(wait)
        ok = len(errors) < attempt
        self.trace.request({**rec, "latency_ms": _ms(t0), **out, "usage": spent or None, "attempts": attempt,
                            **({"errors": errors} if errors else {}), **({} if ok else {"error": errors[-1]})})
        if not ok:
            raise JevError(errors[-1], call_id, filter, attempt, chunks)
        return {q: out["answers"][q] for q in questions}


def _well_formed(a, q: dict) -> bool:
    if q["type"] == "noul":  # a noul answers one probability of yes
        return (isinstance(a, dict) and isinstance(a.get("noul"), (int, float))
                and not isinstance(a["noul"], bool) and 0 <= a["noul"] <= 1)
    if not isinstance(a, dict) or not isinstance(a.get("probabilities"), dict):
        return False
    p = a["probabilities"]  # a choice answers one probability per option
    return set(p) == set(q["criteria"]) and all(isinstance(v, (int, float)) for v in p.values())


def _ms(t0: float) -> int:
    return round((time.monotonic() - t0) * 1000)
