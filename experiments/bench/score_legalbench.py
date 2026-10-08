"""LegalBench-RAG scorer (arXiv:2408.10343), stdlib only. copied into each task dir as bench_score.py.

the answer is one or more <ANSWER> blocks in the final message, each with File: and Quote: lines. each quote is
located in its file (locate()), the located spans are merged per file, then character-level precision and recall
against the gold spans, as the official code defines them (legalbenchrag/run_benchmark.py:16-54): overlap counts only
inside the same file, precision = overlap / predicted chars, recall = overlap / gold chars, 0 when a denominator is 0.
the official code does not merge predicted spans, so overlapping ones count twice; here they are merged first.
whitespace at either end of a gold span is not counted: a quote cannot carry it, and most mini-set snippets have some
(all 453 PrivacyQA ones, 296 of 306 MAUD ones). F1 = 2PR / (P + R)."""
import json
import re
import unicodedata
from pathlib import Path

BLOCK_RX = re.compile(r"<ANSWER>([\s\S]*?)</ANSWER>")
FIELD_RX = re.compile(r"^[\s*>#-]*(File|Quote)\s*\**\s*:\s*\**\s?(.*)$", re.IGNORECASE)
READ_PREFIX = re.compile(r"^\s*\d+\t")
ELLIPSIS = re.compile(r"\s*(?:\.\s?\.\s?\.|\u2026)\s*")
MIN_PART = 12  # a piece of a quote split at "..." must have this many chars (normalized) to be looked up
FOLD = {"\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"', "\u00ab": '"', "\u00bb": '"',
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-"}
QUOTE_MARKS = ('"', "'", "\u201c", "\u201d", "\u2018", "\u2019", "`")


def parse_answers(message: str) -> tuple[list[tuple[str | None, str]], bool]:
    """(file, quote) pairs from every <ANSWER> block, and whether an unclosed last block was read. a quote runs from
    its Quote: line to the next File: or Quote: line or the end of the block; a File: line sets the file for the
    quotes after it"""
    msg = message or ""
    blocks = BLOCK_RX.findall(msg)
    unclosed = False
    tail = msg.rfind("<ANSWER>")
    if tail >= 0 and msg.find("</ANSWER>", tail) < 0:
        blocks.append(msg[tail + len("<ANSWER>"):])
        unclosed = True
    pairs = []
    for b in blocks:
        cur_file, cur = None, None
        for line in b.split("\n"):
            m = FIELD_RX.match(line)
            if m and m.group(1).lower() == "file":
                if cur is not None:
                    pairs.append((cur_file, "\n".join(cur)))
                    cur = None
                cur_file = m.group(2).strip().strip("`*\"'").strip() or None
            elif m and m.group(1).lower() == "quote":
                if cur is not None:
                    pairs.append((cur_file, "\n".join(cur)))
                cur = [m.group(2)]
            elif cur is not None:
                cur.append(line)
        if cur is not None:
            pairs.append((cur_file, "\n".join(cur)))
    return [(f, clean_quote(q)) for f, q in pairs if clean_quote(q)], unclosed


def clean_quote(q: str) -> str:
    """trim; take off read-tool line numbers when every line has one; take off one pair of outer quote marks"""
    q = q.strip()
    rows = q.split("\n")
    body = [r for r in rows if r.strip()]
    if body and all(READ_PREFIX.match(r) for r in body):
        q = "\n".join(READ_PREFIX.sub("", r) for r in rows).strip()
    if len(q) >= 2 and q[0] in QUOTE_MARKS and q[-1] in QUOTE_MARKS:
        q = q[1:-1].strip()
    return q


def normalize(text: str) -> tuple[str, list[int]]:
    """text with curly quotes and dashes folded to ascii and every run of whitespace made one space, plus the
    index in text of each kept char"""
    out, idx = [], []
    for i, ch in enumerate(text):
        c = FOLD.get(ch, ch)
        if c.isspace():
            if out and out[-1] == " ":
                continue
            c = " "
        out.append(c)
        idx.append(i)
    return "".join(out), idx


def _find(doc: str, ndoc: tuple[str, list[int]], q: str, start: int = 0):
    """(s, e, how) of the first match at or after char start: exact, else normalized. None if not found"""
    i = doc.find(q, start)
    if i >= 0:
        return i, i + len(q), "exact"
    nd, idx = ndoc
    nq = normalize(unicodedata.normalize("NFC", q))[0].strip()
    if not nq:
        return None
    k0 = next((k for k, v in enumerate(idx) if v >= start), len(idx))
    j = nd.find(nq, k0)
    if j < 0:
        return None
    return idx[j], idx[j + len(nq) - 1] + 1, "normalized"


def locate(doc: str, q: str, ndoc=None) -> tuple[list[tuple[int, int]], str]:
    """spans of a quote in its document and how they were found: exact first occurrence; else the first match after
    normalizing both; else the quote split at "..." with each piece of MIN_PART or more chars found in order;
    else nothing (unmatched)"""
    ndoc = ndoc or normalize(doc)
    hit = _find(doc, ndoc, q)
    if hit:
        return [(hit[0], hit[1])], hit[2]
    parts = [p for p in ELLIPSIS.split(q) if len(normalize(p)[0].strip()) >= MIN_PART]
    if len(parts) > 1:
        spans, at = [], 0
        for p in parts:
            h = _find(doc, ndoc, p.strip(), at)
            if h:
                spans.append((h[0], h[1]))
                at = h[1]
        if spans:
            return spans, "ellipsis"
    return [], "unmatched"


def merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def overlap(a: list, b: list) -> int:
    n = 0
    for s, e in merge(a):
        for s2, e2 in merge(b):
            n += max(0, min(e, e2) - max(s, s2))
    return n


def char_scores(pred: dict, gold: dict) -> dict:
    """pred and gold: file -> [(s, e)]. precision, recall and F1 on characters"""
    hit = sum(overlap(pred.get(f, []), gold.get(f, [])) for f in set(pred) | set(gold))
    np_ = sum(e - s for v in pred.values() for s, e in merge(v))
    ng = sum(e - s for v in gold.values() for s, e in merge(v))
    p = hit / np_ if np_ else 0.0
    r = hit / ng if ng else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "hit": hit, "pred": np_,
            "gold": ng}


def resolve(name: str | None, files: list[str]) -> str:
    """a File: value to a document: the path itself, else its base name, else a unique case-insensitive base-name
    match, else a unique file that ends with it. "" if none"""
    if not name:
        return ""
    n = unicodedata.normalize("NFC", name.strip().replace("\\", "/")).lstrip("./")
    if n in files:
        return n
    base = n.rsplit("/", 1)[-1]
    if base in files:
        return base
    low = [f for f in files if f.lower() == base.lower()]
    if len(low) == 1:
        return low[0]
    ends = [f for f in files if n.endswith(f) or f.endswith(base)]
    return ends[0] if len(ends) == 1 else ""


def trim(text: str, s: int, e: int) -> tuple[int, int]:
    """a span without the whitespace at its ends"""
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def read_doc(p: Path) -> str:
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def score(task_dir: Path, work: Path, run_dir: Path | None, spec: dict) -> dict:
    repo = task_dir / "repo"
    files = sorted(p.name for p in repo.iterdir() if p.is_file())
    result = json.loads((run_dir / "result.json").read_text())
    pairs, unclosed = parse_answers(result.get("final_message") or "")
    docs, pred, rows = {}, {}, []
    for f, q in pairs:
        rel = resolve(f, files)
        if not rel:
            rows.append({"file": f, "quote": q[:200], "how": "no such file"})
            continue
        if rel not in docs or docs[rel][1] is None:
            text = read_doc(repo / rel)
            docs[rel] = (text, normalize(text))
        spans, how = locate(docs[rel][0], q, docs[rel][1])
        pred.setdefault(rel, []).extend(spans)
        rows.append({"file": rel, "quote": q[:200], "how": how, "spans": spans})
    gold = {}
    for g in spec["gold"]:
        if g["file"] not in docs:
            text = read_doc(repo / g["file"])
            docs[g["file"]] = (text, None)
        gold.setdefault(g["file"], []).append(trim(docs[g["file"]][0], *g["span"]))
    s = char_scores(pred, gold)
    out = {f"lb_{k}": v for k, v in s.items()}
    out["lb_quotes"] = len(pairs)
    out["lb_quotes_unmatched"] = sum(1 for r in rows if r["how"] in ("unmatched", "no such file"))
    out["lb_quotes_normalized"] = sum(1 for r in rows if r["how"] in ("normalized", "ellipsis"))
    out["lb_answer_found"] = bool(pairs)
    out["lb_answer_unclosed"] = unclosed
    detail = (f"quotes {len(pairs)} ({out['lb_quotes_unmatched']} unmatched); chars hit {s['hit']} of gold "
              f"{s['gold']}, predicted {s['pred']}")
    if not pairs:
        detail = "no <ANSWER> block with a Quote"
    return {"score": s["f1"], "detail": detail, **out, "_extra": {"quotes": rows}}
