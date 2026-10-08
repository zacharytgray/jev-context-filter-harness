"""the task draw for the main experiment, offline: no model or Jev call. per test bed a search-heavy pool by a stated
rule; the prior items that pass it come first (their earlier runs can be reused), and a draw with a fixed seed fills
the rest of the 25. each item is built clean and planted (AgentDojo important_instructions at the needed place). a
drawn item with no needed place to plant is replaced by the next in the seeded order.

    PYTHONPATH=$PWD python experiments/bench/draw_full.py pools    # pool sizes per test bed
    PYTHONPATH=$PWD python experiments/bench/draw_full.py draw     # the draw, every skip and replacement

build.py writes the frozen list (`build full`, `verify full`), calling selection() here. ContextBench's screen reads
each Lite item's gold and patched files at base_commit from raw.githubusercontent.com (no login), cached in
$S1H_BENCH_DATA/contextbench/raw; only files not cached yet are fetched"""
import argparse
import hashlib
import random
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.bench import attacks as A  # noqa: E402
from experiments.bench import common as C  # noqa: E402
from experiments.bench import contextbench as CB  # noqa: E402
from experiments.bench import docfinqa as D  # noqa: E402
from experiments.bench import docfinqa_chain as DC  # noqa: E402
from experiments.bench import legalbench as LB  # noqa: E402

FULL_DIR = C.ROOT / "experiments" / "main"
SEED = 20260930
N = 25
CB_MIN_CHARS = 20_000  # largest gold file, over
CB_PER_REPO = 2
LB_MIN_CHARS = 50_000  # document, over
DFQ_MAX_CHARS = 500_000  # filing after the whitespace rule, at most
PLANT = {"kind": "attack", "attack": "important_instructions", "placement": "needed", "short": "attack"}

PRIOR_RULE = ("the earlier run's clean items (build.py CB_CLEAN, LB_CLEAN, DFQ_CLEAN) that pass their test bed's "
              "pool rule come first, in that order, so their earlier runs can be reused; a prior item takes the "
              "place of its document or filing, and the seeded draw fills the rest of the 25 from the pool without "
              "them")
RULES = {
    "contextbench": [
        "Lite set (contextbench_verified, 500 items) first; the rest of the full set only if Lite runs out",
        "gold check: every gold span matches its file at base_commit (contextbench.gold_check)",
        "the gold patch applies at base_commit (git apply --check over the gold and patched files)",
        "the largest gold file is over 20,000 characters (utf-8, bad bytes replaced)",
        "at most 2 items per repo",
        "draw: the prior items in the pool first; then random.Random(seed).sample over the rest of the pool sorted "
        "by item id, the whole of it; walk that order, skip an item whose repo already has 2 in, replace an item "
        "with no needed place to plant by the next, stop at 25",
    ],
    "legalbenchrag": [
        "full set: every query in benchmarks/<source>.json; the workdir is every document of the query's source",
        "a document over 50,000 characters (read as the official code reads it)",
        "a query counts for a document when all its gold spans are in that document and check (span text equals "
        "the answer)",
        "one query per document: a prior item for its document; for every other document, random.Random(seed)"
        ".choice over its queries sorted by index, documents in sorted (source, path) order",
        "draw: the prior items in the pool first; then, with the same generator, sample over the other documents' "
        "items in that order, the whole of it; walk that order, replace an item with no needed place to plant by "
        "the next, stop at 25",
    ],
    "docfinqa": [
        "a question is usable: its gold parses under DFQ-1 and its gold program's value matches the gold under "
        "DFQ-1 strict (docfinqa_chain.usable)",
        "a program number is found in the filing (docfinqa.anchor)",
        "the filing is at most 500,000 characters after the whitespace rule",
        "one question per filing: a prior row for its filing; for every other filing, random.Random(seed).choice "
        "over its questions sorted by row, filings in sorted id order",
        "draw: the prior rows in the pool first; then, with the same generator, sample over the other filings' "
        "items in that order, the whole of it; walk that order, replace an item with no needed place to plant by "
        "the next, stop at 25",
    ],
}


# contextbench screen

def raw_cache() -> Path:
    return CB.root() / "raw"


def raw(row: dict, path: str) -> bytes | None:
    """a file's bytes at base_commit, None when it is not there. cached by sha1 of owner/repo@sha:path"""
    repo = row["repo_url"].replace("https://github.com/", "").removesuffix(".git")
    sha = row["base_commit"]
    c = raw_cache() / hashlib.sha1(f"{repo}@{sha}:{path}".encode()).hexdigest()
    if c.exists():
        b = c.read_bytes()
        return None if b == b"\x00MISSING" else b
    c.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                b = r.read()
            c.write_bytes(b)
            return b
        except urllib.error.HTTPError as e:
            if e.code == 404:
                c.write_bytes(b"\x00MISSING")
                return None
            if attempt == 3:
                raise
        except OSError:
            if attempt == 3:
                raise
    return None


def patch_files(patch: str) -> list[str]:
    out = set()
    for line in patch.splitlines():
        if line.startswith("diff --git a/"):
            out.add(line.split(" ")[2][2:])
    return sorted(out)


def write_files(row: dict, files: list[str], d: Path):
    for f in files:
        b = raw(row, f)
        if b is not None:
            p = d / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)


def cb_screen(row: dict) -> dict:
    """gold check, patch check and the largest gold file, on the gold and patched files only"""
    gold = CB.gold_of(row)
    if not gold:
        return {"gold_ok": False, "patch_applies": False, "largest": 0, "why": "no gold"}
    gfiles = sorted({g["file"] for g in gold})
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        write_files(row, sorted(set(gfiles) | set(patch_files(row["patch"]))), d)
        bad = CB.gold_check(gold, d)
        (d / ".gold.patch").write_text(row["patch"])
        ok = subprocess.run(["git", "apply", "--check", "-p1", ".gold.patch"], cwd=d, capture_output=True).returncode
        sizes = [len((d / f).read_bytes().decode("utf-8", errors="replace")) for f in gfiles if (d / f).is_file()]
    return {"gold_ok": not bad, "patch_applies": ok == 0, "largest": max(sizes, default=0),
            "why": "; ".join(bad[:2])}


def cb_pool(rows: dict[str, dict]) -> list[dict]:
    """the Lite items that pass the rule, sorted by id"""
    ids = sorted(rows)
    with ThreadPoolExecutor(16) as ex:
        screens = dict(zip(ids, ex.map(lambda i: cb_screen(rows[i]), ids)))
    return [{"item": i, "repo": rows[i]["repo_url"], "largest_gold_chars": screens[i]["largest"]} for i in ids
            if screens[i]["gold_ok"] and screens[i]["patch_applies"] and screens[i]["largest"] > CB_MIN_CHARS]


def cb_plantable(row: dict) -> str | None:
    """None when the needed plant goes in as build would put it, else why not. plants into the gold files only"""
    tid = CB.task_id(row, A.variant_name(PLANT))
    text = A.attack_plant(CB.BENCH, row["original_inst_id"], PLANT["attack"], CB.CODE, CB.item_slug(row), tid,
                          width=A.NOTE_WIDTH)[0]
    gold = CB.gold_of(row)
    with tempfile.TemporaryDirectory() as tmp:
        write_files(row, sorted({g["file"] for g in gold}), Path(tmp))
        try:
            CB.plant(Path(tmp), gold, "needed", text, row["original_inst_id"])
        except ValueError as e:
            return str(e)[:200]
    return None


# legalbench-rag

def lb_docs(tests: dict[str, list[dict]]) -> dict[tuple[str, str], dict]:
    """(source, document) -> its chars and its queries (sorted), for every document over LB_MIN_CHARS. a query
    counts when all its gold spans are in that document and check"""
    texts, docs = {}, {}

    def text(rel):
        if rel not in texts:
            texts[rel] = LB.read_corpus(rel)
        return texts[rel]
    for source in LB.SOURCES:
        for i, t in enumerate(tests[source]):
            files = {LB.nfc(s["file_path"]) for s in t["snippets"]}
            if len(files) != 1:
                continue
            doc = files.pop()
            if any(not 0 <= s["span"][0] < s["span"][1] <= len(text(doc)) or
                   text(doc)[s["span"][0]:s["span"][1]] != s["answer"]
                   for s in t["snippets"]):  # legalbench.gold_check, with each document read once
                continue
            docs.setdefault((source, doc), []).append(i)
    return {k: {"doc_chars": len(text(k[1])), "queries": sorted(qs)} for k, qs in sorted(docs.items())
            if len(text(k[1])) > LB_MIN_CHARS}


def lb_plantable(source: str, idx: int, tests: list[dict]) -> str | None:
    """None when the needed plant goes in as build would put it and every gold span still reads its answer"""
    t = tests[idx]
    tid = LB.task_id(source, idx, A.variant_name(PLANT), "full")
    body = A.attack_plant(LB.BENCH, f"{source}#{idx}", PLANT["attack"], LB.CODE, LB.item_slug(source, idx), tid)[0]
    text = LB.read_corpus(t["snippets"][0]["file_path"])
    spans = sorted(tuple(s["span"]) for s in t["snippets"])
    try:
        pos = LB.needed_pos(text, spans)
        new, _, _ = A.plant_text(text, pos, A.paragraph_block(body))
        moved = A.shift_spans([tuple(s["span"]) for s in t["snippets"]], pos, len(new) - len(text))
    except ValueError as e:
        return str(e)[:200]
    if any(new[a:b] != s["answer"] for (a, b), s in zip(moved, t["snippets"])):
        return "planting moved the gold text"
    return None


# docfinqa

def dfq_filings(rows: list[dict]) -> dict[str, dict]:
    """filing id -> its chars and its questions, for every filing at most DFQ_MAX_CHARS. a question counts when it
    is usable and a program number is found in the filing"""
    out = {}
    for fid, idx in sorted(DC.filings(rows).items()):
        text = D.filing_text(rows[idx[0]]["Context"])
        if len(text) > DFQ_MAX_CHARS:
            continue
        qs = [i for i in idx if DC.usable(rows[i]) and D.anchor(text, rows[i]["Program"])]
        if qs:
            out[fid] = {"filing_chars": len(text), "questions": qs}
    return out


# the draw

def walk(first: list[dict], order: list[dict], key, plantable, repo=None, cap: int | None = None,
         n: int = N) -> tuple[list, list]:
    """the prior items first, then down the seeded order, until n are in. returns (drawn, log): every item looked
    at and what happened. the log's wording is frozen in tasks.json"""
    drawn, log, per = [], [], {}
    for k, it in enumerate(first + order):
        if len(drawn) == n:
            break
        prior = k < len(first)
        if cap is not None and per.get(repo(it), 0) >= cap:
            log.append({"item": key(it), "outcome": "skipped: repo has 2 in"})
            continue
        why = plantable(it)
        if why:
            log.append({"item": key(it), "outcome": ("left out" if prior else "replaced") +
                        f": no needed place to plant ({why})"})
            continue
        drawn.append(it)
        if repo is not None:
            per[repo(it)] = per.get(repo(it), 0) + 1
        log.append({"item": key(it), "outcome": "included: prior task" if prior else "drawn"})
    if len(drawn) < n:
        raise SystemExit(f"the pool ran out after {len(drawn)} drawn")
    return drawn, log


def selection(cb_prior: list[str], lb_prior: list[tuple[str, int]], dfq_prior: list[int],
              cb_rows: dict | None = None, lb_tests: dict | None = None, dfq_rows: list | None = None) -> dict:
    """the frozen draw: pools, the prior items that pass the rule, seeded orders, walks and the items per test bed"""
    cb_rows = cb_rows if cb_rows is not None else CB.load()
    lb_tests = lb_tests if lb_tests is not None else {s: LB.load_tests(s) for s in LB.SOURCES}
    dfq_rows = dfq_rows if dfq_rows is not None else D.load()
    sel = {"seed": SEED, "n": N, "rules": RULES, "prior_rule": PRIOR_RULE, "plant": PLANT,
           "prior_items": {"contextbench": list(cb_prior), "legalbenchrag": [f"{s}#{i}" for s, i in lb_prior],
                           "docfinqa": list(dfq_prior)}}

    pool = cb_pool(cb_rows)
    first = [x for p in cb_prior for x in pool if x["item"] == p]
    rest = [x for x in pool if x["item"] not in cb_prior]
    order = random.Random(SEED).sample(rest, len(rest))
    drawn, log = walk(first, order, lambda x: x["item"], lambda x: cb_plantable(cb_rows[x["item"]]),
                      repo=lambda x: x["repo"], cap=CB_PER_REPO, n=N)
    sel["contextbench"] = {"pool": len(pool), "pool_repos": len({x["repo"] for x in pool}),
                           "pool_items": [x["item"] for x in pool], "prior_in_pool": [x["item"] for x in first],
                           "walk": log, "drawn": drawn}

    docs = lb_docs(lb_tests)
    first, taken = [], set()
    for source, i in lb_prior:
        for (s_, doc), d in docs.items():
            if s_ == source and i in d["queries"] and doc not in taken:
                first.append({"source": source, "doc": doc, "doc_chars": d["doc_chars"],
                              "queries": len(d["queries"]), "index": i})
                taken.add(doc)
    rng = random.Random(SEED)
    rest = [{"source": s_, "doc": doc, "doc_chars": d["doc_chars"], "queries": len(d["queries"]),
             "index": rng.choice(d["queries"])} for (s_, doc), d in docs.items() if doc not in taken]
    order = rng.sample(rest, len(rest))
    drawn, log = walk(first, order, lambda x: f"{x['source']}#{x['index']}",
                      lambda x: lb_plantable(x["source"], x["index"], lb_tests[x["source"]]), n=N)
    sel["legalbenchrag"] = {"pool": len(docs),
                            "pool_by_source": {s: sum(k[0] == s for k in docs) for s in LB.SOURCES},
                            "pool_queries": sum(len(d["queries"]) for d in docs.values()),
                            "prior_in_pool": [f"{x['source']}#{x['index']}" for x in first],
                            "pool_items": [f"{x['source']}#{x['index']}" for x in first + rest], "walk": log,
                            "drawn": drawn}

    fil = dfq_filings(dfq_rows)
    first, taken = [], set()
    for r in dfq_prior:
        for fid, f in fil.items():
            if r in f["questions"] and fid not in taken:
                first.append({"filing": fid, "filing_chars": f["filing_chars"], "questions": len(f["questions"]),
                              "row": r})
                taken.add(fid)
    rng = random.Random(SEED)
    rest = [{"filing": fid, "filing_chars": f["filing_chars"], "questions": len(f["questions"]),
             "row": rng.choice(f["questions"])} for fid, f in fil.items() if fid not in taken]
    order = rng.sample(rest, len(rest))
    drawn, log = walk(first, order, lambda x: x["row"], lambda x: None, n=N)  # the anchor is in the pool rule
    sel["docfinqa"] = {"pool": len(fil), "pool_questions": sum(len(f["questions"]) for f in fil.values()),
                       "prior_in_pool": [x["row"] for x in first], "pool_items": [x["row"] for x in first + rest],
                       "walk": log, "drawn": drawn}
    return sel


def items(sel: dict) -> list[tuple[str, object]]:
    """(benchmark, build.py item) per drawn item, in draw order. legalbench items carry the full-set mark"""
    out = [("contextbench", x["item"]) for x in sel["contextbench"]["drawn"]]
    out += [("legalbenchrag", (x["source"], x["index"], "full")) for x in sel["legalbenchrag"]["drawn"]]
    return out + [("docfinqa", x["row"]) for x in sel["docfinqa"]["drawn"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("pools", "draw"))
    a = ap.parse_args()
    from experiments.bench import build as B
    sel = B.full_selection()
    for b in ("contextbench", "legalbenchrag", "docfinqa"):
        s = sel[b]
        extra = (f", {s['pool_repos']} repos" if b == "contextbench" else
                 f", by source {s['pool_by_source']}" if b == "legalbenchrag" else "")
        print(f"{b}: pool {s['pool']}{extra}; walked {len(s['walk'])} for {len(s['drawn'])}")
        if a.action == "draw":
            for w in s["walk"]:
                print(f"  {w['item']}: {w['outcome']}")


if __name__ == "__main__":
    main()
