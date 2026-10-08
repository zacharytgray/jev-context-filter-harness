"""DocFinQA scorer (arXiv:2401.06915), stdlib only. copied into each task dir as bench_score.py.

rule DFQ-1 (the paper states no tolerance; see experiments/bench/README.md for the reasons and examples):
1. extract: the last line that starts with "ANSWER:" (any case; spaces, *, #, > or - may come first), from after the
   colon to the end of the line, with markdown marks (* `) and spaces trimmed. with no such line, the text after the
   last "ANSWER:" anywhere, to the end of its line.
2. parse (both sides): lowercase, trim, U+2212 read as "-". an answer that starts with the word yes or no is a yes/no
   answer. otherwise the leading number: an optional "$", an optional sign, digits with commas, an optional decimal
   point and decimals, an optional exponent (1e3); "(x)", "(x%)" and "(x)%" mean -x (an unclosed "(" is ignored);
   a "%" or the word "percent" or "per cent" right after the number marks a percent. anything after it (unit words,
   ":1") is ignored. a gold that does not parse is not scored.
3. pre-round the prediction to 9 decimals, half-up (removes float noise like 50.724999999999994). decimal math runs
   with 80 digits; a prediction too large even for that is wrong, not an error.
4. candidates: the prediction p; also 100p when the gold has % and the prediction does not; also p/100 when the
   prediction has % and the gold does not.
5. strict (headline): d = the number of decimals written in the gold. correct when some candidate, rounded half-up
   or truncated toward zero to d decimals, equals the gold. yes/no must match exactly.
6. close (secondary, BizBench's "within 1%"): strict, or |c - g| <= 0.01 |g| for some candidate (|c| <= 0.005 when
   g = 0).
7. the sign must match; a prediction off only by sign is reported as sign_flip, never as correct."""
import json
import re
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from pathlib import Path

ANSWER_RX = re.compile(r"answer\s*\**\s*:", re.IGNORECASE)
ANSWER_LINE = re.compile(r"^[\s*#>-]*answer\s*\**\s*:(.*)$", re.IGNORECASE | re.MULTILINE)
NUM_RX = re.compile(r"^\$?\s*(?P<open>\()?\s*\$?\s*(?P<sign>[+-])?\s*\$?\s*(?P<num>\d[\d,]*(?:\.\d*)?|\.\d+)"
                    r"(?:e(?P<exp>[+-]?\d+))?\s*(?P<close1>\))?\s*(?P<pct>%|percent\b|per cent\b)?\s*(?P<close2>\))?")
YES_NO = re.compile(r"^(yes|no)\b")


def extract(message: str) -> str | None:
    """the answer text of the last ANSWER: line, else after the last ANSWER: anywhere. None when there is none"""
    msg = message or ""
    lines = list(ANSWER_LINE.finditer(msg))
    if lines:
        rest = lines[-1].group(1)
    else:
        hits = list(ANSWER_RX.finditer(msg))
        if not hits:
            return None
        rest = msg[hits[-1].end():].split("\n", 1)[0]
    return rest.strip(" \t*`")


def parse(s: str):
    """("yn", "yes"|"no") or ("num", Decimal, is_percent, decimals). None when it does not parse"""
    if s is None:
        return None
    s = s.strip().lower().replace("\u2212", "-")
    m = YES_NO.match(s)
    if m:
        return ("yn", m.group(1))
    m = NUM_RX.match(s)
    if not m:
        return None
    num = m.group("num").replace(",", "")
    try:
        v = Decimal(num + (f"e{m.group('exp')}" if m.group("exp") else ""))
    except InvalidOperation:
        return None
    if m.group("sign") == "-" or (m.group("open") and (m.group("close1") or m.group("close2"))):
        v = -v
    d = len(num.split(".")[1]) if "." in num else 0
    return ("num", v, m.group("pct") is not None, d)


def judge(pred: str | None, gold: str) -> dict:
    """{"scorable", "strict", "close", "sign_flip", "parsed"} for one prediction against one gold answer"""
    G, P = parse(gold), parse(pred)
    out = {"scorable": G is not None, "strict": False, "close": False, "sign_flip": False, "parsed": P is not None}
    if G is None or P is None:
        return out
    if G[0] == "yn" or P[0] == "yn":
        ok = G[0] == P[0] == "yn" and G[1] == P[1]
        return {**out, "strict": ok, "close": ok}
    _, g, gp, d = G
    _, p, pp, _ = P
    with localcontext() as ctx:
        ctx.prec = 80
        try:
            p = p.quantize(Decimal("1e-9"), ROUND_HALF_UP)
            cands = [p]
            if gp and not pp:
                cands.append(p * 100)
            if pp and not gp:
                cands.append(p / 100)
            q = Decimal(1).scaleb(-d)

            def strict_ok(c):
                return c.quantize(q, ROUND_HALF_UP) == g or c.quantize(q, ROUND_DOWN) == g

            def close_ok(c):
                return abs(c - g) <= Decimal("0.01") * abs(g) if g != 0 else abs(c) <= Decimal("0.005")
            strict = any(strict_ok(c) for c in cands)
            close = strict or any(close_ok(c) for c in cands)
            flip = not close and g != 0 and any(strict_ok(-c) or close_ok(-c) for c in cands)
        except InvalidOperation:  # a number too large to round: it cannot equal the gold
            return out
    return {**out, "strict": strict, "close": close, "sign_flip": flip}


def score(task_dir: Path, work: Path, run_dir: Path | None, spec: dict) -> dict:
    result = json.loads((run_dir / "result.json").read_text())
    pred = extract(result.get("final_message") or "")
    j = judge(pred, spec["answer"])
    out = {"dfq_answer": pred, "dfq_gold": spec["answer"], "dfq_strict": j["strict"], "dfq_close": j["close"],
           "dfq_sign_flip": j["sign_flip"], "dfq_parsed": j["parsed"], "dfq_scorable": j["scorable"]}
    if not j["scorable"]:
        return {"score": None, "detail": f"gold {spec['answer']!r} is not scorable", **out}
    detail = f"answer {pred!r}, gold {spec['answer']!r}: strict {j['strict']}, close {j['close']}"
    if pred is None:
        detail = "no ANSWER: line"
    return {"score": 1.0 if j["strict"] else 0.0, "detail": detail, **out}
