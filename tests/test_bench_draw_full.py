"""the main experiment's task draw (experiments/bench/draw_full.py): the walk (prior items first, the repo cap, a
replacement), the legalbench document pool, the docfinqa filing pool, the whole selection on tiny fixtures with the
contextbench screen stubbed, and the list build.py makes from it. no network, no benchmark data"""
import json
import random

import pytest

from experiments.bench import build as B
from experiments.bench import docfinqa_chain as Q
from experiments.bench import draw_full as F
from experiments.bench import legalbench as LB


def it(name, repo="r"):
    return {"item": name, "repo": repo}


def test_walk_takes_prior_items_first_caps_repos_and_replaces():
    first = [it("p1", "a")]
    order = [it("x1", "a"), it("x2", "a"), it("bad", "b"), it("x3", "b"), it("x4", "c")]
    drawn, log = F.walk(first, order, lambda x: x["item"], lambda x: "no spot" if x["item"] == "bad" else None,
                        repo=lambda x: x["repo"], cap=2, n=4)
    assert [x["item"] for x in drawn] == ["p1", "x1", "x3", "x4"]
    assert [w["outcome"].split(":")[0] for w in log] == ["included", "drawn", "skipped", "replaced", "drawn", "drawn"]
    # the outcome wording is frozen in tasks.json's selection
    assert log[0]["outcome"] == "included: prior task" and "no spot" in log[3]["outcome"]
    with pytest.raises(SystemExit):
        F.walk([], order[:2], lambda x: x["item"], lambda x: None, n=3)


def test_walk_leaves_out_a_prior_item_with_no_spot():
    drawn, log = F.walk([it("p1")], [it("x1"), it("x2")], lambda x: x["item"],
                        lambda x: "none" if x["item"] == "p1" else None, n=2)
    assert [x["item"] for x in drawn] == ["x1", "x2"] and log[0]["outcome"].startswith("left out")


BIG = 60_000


def lb_fixture(tmp_path, monkeypatch):
    """maud: two big documents (a: 3 queries, b: 1) and a small one. cuad: one big document whose only query has a
    span that does not check, and a query across two documents"""
    monkeypatch.setenv("S1H_BENCH_DATA", str(tmp_path))
    corpus = tmp_path / "legalbenchrag" / "data" / "corpus"
    docs = {"maud/a.txt": "A" * 100 + "gold a words" + "A" * BIG, "maud/b.txt": "gold b" + "B" * BIG,
            "maud/small.txt": "gold s" + "S" * 100, "cuad/c.txt": "C" * BIG + "gold c",
            "cuad/d.txt": "D" * BIG}
    for rel, text in docs.items():
        (corpus / rel).parent.mkdir(parents=True, exist_ok=True)
        (corpus / rel).write_text(text)

    def q(rel, s, e, answer=None):
        return {"query": f"q {rel}", "snippets": [{"file_path": rel, "span": [s, e],
                                                   "answer": answer or docs[rel][s:e]}]}
    tests = {"maud": [q("maud/a.txt", 100, 112), q("maud/b.txt", 0, 6), q("maud/a.txt", 0, 10),
                      q("maud/small.txt", 0, 6), q("maud/a.txt", 104, 112)],
             "cuad": [q("cuad/c.txt", BIG, BIG + 6, "wrong!"),
                      {"query": "two docs", "snippets": q("cuad/c.txt", 0, 5)["snippets"] +
                       q("cuad/d.txt", 0, 5)["snippets"]}],
             "contractnli": [], "privacy_qa": []}
    bdir = tmp_path / "legalbenchrag" / "data" / "benchmarks"
    bdir.mkdir(parents=True)
    for s, t in tests.items():
        (bdir / f"{s}.json").write_text(json.dumps({"tests": t}))
    return tests


def test_lb_docs_keeps_big_documents_with_checked_single_document_queries(tmp_path, monkeypatch):
    tests = lb_fixture(tmp_path, monkeypatch)
    docs = F.lb_docs(tests)
    assert list(docs) == [("maud", "maud/a.txt"), ("maud", "maud/b.txt")]
    assert docs[("maud", "maud/a.txt")]["queries"] == [0, 2, 4]
    assert LB.full_docs("maud", tests["maud"]) == ["maud/a.txt", "maud/b.txt", "maud/small.txt"]


def dfq_fixture() -> list[dict]:
    """filings: A with two usable plantable questions, B with one, C too big, D with none usable"""
    def ctx(tag, nums, pad=0):
        return f"Report {tag}.\n\n" + "\n".join(f"| item {k} | {x} |" for k, x in enumerate(nums)) + "\n\nEnd." + \
            " filler" * pad + "\n"

    def row(c, a, b, gold=None):
        return {"Context": c, "Question": f"q {a}", "Program": f"a = {a}\nb = {b}\nanswer = a + b",
                "Answer": gold or f"{a + b:.2f}"}
    ca, cb, cc, cd = ctx("A", [101.5, 202.25]), ctx("B", [303.75]), ctx("C", [404.5], pad=90_000), ctx("D", [5.5])
    return [row(ca, 101.5, 7.25), row(cb, 303.75, 1.25), row(ca, 202.25, 1.5), row(cc, 404.5, 1.5),
            row(cd, 5.5, 1.5, gold="see note")]


def test_dfq_filings_keep_usable_plantable_questions_under_the_cap(monkeypatch):
    monkeypatch.setattr(F, "DFQ_MAX_CHARS", 500_000)
    rows = dfq_fixture()
    fil = F.dfq_filings(rows)
    assert sorted(f["questions"] for f in fil.values()) == [[0, 2], [1]]
    assert Q.filing_id(rows[3]["Context"]) not in fil


def stub_cb(monkeypatch, pool):
    monkeypatch.setattr(F, "cb_pool", lambda rows: pool)
    monkeypatch.setattr(F, "cb_plantable", lambda row: None)


def test_selection_puts_prior_items_first_and_is_seeded(tmp_path, monkeypatch):
    tests = lb_fixture(tmp_path, monkeypatch)
    rows = dfq_fixture()
    pool = [{"item": f"o__r{k}-{j}", "repo": f"r{k}", "largest_gold_chars": 30_000} for k in range(20)
            for j in range(3)]
    stub_cb(monkeypatch, pool)
    monkeypatch.setattr(F, "N", 2)

    def sel():
        return F.selection(["o__r3-1", "missing"], [("maud", 4), ("maud", 2)], [2, 3],
                           cb_rows={x["item"]: {} for x in pool}, lb_tests=tests, dfq_rows=rows)
    s = sel()
    assert s == sel() and json.loads(json.dumps(s)) == s
    assert s["contextbench"]["prior_in_pool"] == ["o__r3-1"] and s["contextbench"]["drawn"][0]["item"] == "o__r3-1"
    rest = [x for x in pool if x["item"] != "o__r3-1"]
    assert s["contextbench"]["drawn"][1] == random.Random(F.SEED).sample(rest, len(rest))[0]
    # maud#4 takes document a; maud#2 is on the same document, so it is not a second prior item
    lb = s["legalbenchrag"]
    assert lb["prior_in_pool"] == ["maud#4"] and lb["pool"] == 2
    assert [(x["source"], x["index"]) for x in lb["drawn"]] == [("maud", 4), ("maud", 1)]
    dfq = s["docfinqa"]
    assert dfq["prior_in_pool"] == [2] and [x["row"] for x in dfq["drawn"]] == [2, 1]


def test_full_list_builds_each_item_clean_and_planted():
    sel = {"contextbench": {"drawn": [{"item": "psf__requests-1142"}]},
           "legalbenchrag": {"drawn": [{"source": "maud", "index": 1166}]},
           "docfinqa": {"drawn": [{"row": 624}]}}
    lst = B.full_list(sel)
    ids = [B.tid(b, i, v) for b, i, v in lst]
    assert ids == ["cb-psf-requests-1142-clean", "cb-psf-requests-1142-attack-needed", "lbf-maud-1166-clean",
                   "lbf-maud-1166-attack-needed", "dfq-624-clean", "dfq-624-attack-needed"]
    assert [v for _, _, v in lst[:2]] == [B.clean(), B.planted_attack("needed")]
    assert B.item_id("legalbenchrag", ("maud", 1166, "full")) == "maud#1166"
    assert B.list_file("full") == "tasks.json" and B.list_dir("full") == F.FULL_DIR

