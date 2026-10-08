"""the DocFinQA filing grouping and usable-question rule the task draw reads (docfinqa_chain.py), the task dir
writer's extra files, and build.py's list writer and links. tiny fixtures, no network, no benchmark data"""
import json
import shutil
from pathlib import Path

import pytest

from experiments.bench import build as B
from experiments.bench import common as C
from experiments.bench import docfinqa_chain as Q


def ctx(tag: str, nums: list, pad: int = 0) -> str:
    parts = [f"Annual report {tag}."]
    for j, x in enumerate(nums):
        parts.append(f"Item {j} of {tag} shows the figures below.\n| label {j} | {x} |")
    parts.append("Closing remarks." + " filler" * pad)
    return "\n\n".join(parts) + "\n"


def row(context: str, question: str, a: float, b: float, gold: str | None = None) -> dict:
    return {"Context": context, "Question": question, "Program": f"a = {a}\nb = {b}\nanswer = a + b",
            "Answer": gold if gold is not None else f"{a + b:.2f}"}


A_NUMS = [101.5, 202.25, 303.75, 404.5, 505.25, 606.5]
B_NUMS = [11.5, 22.25, 33.75, 44.5, 55.25]

def fixture() -> list[dict]:
    """seven filings: A with 6 questions, B to G with 5 each. F has one unscorable gold"""
    rows = [row(ctx("D", B_NUMS), f"d{j}", B_NUMS[j], 1.25) for j in range(5)]  # rows 0-4, excluded by row 0
    ca = ctx("A", A_NUMS)
    rows += [row(ca, f"a question {j}", A_NUMS[j], A_NUMS[(j + 1) % 6]) for j in range(6)]
    cb = ctx("B", B_NUMS)
    rows += [row(cb, "b question 0", B_NUMS[0], B_NUMS[1])]
    rows += [row(cb, f"b question {j}", 7001.5 + j, 8002.25) for j in range(1, 5)]
    cc = ctx("C", B_NUMS, pad=400)
    rows += [row(cc, f"c{j}", B_NUMS[j], 2.5) for j in range(5)]
    ce = ctx("E", B_NUMS)
    rows += [row(ce, "same question" if j < 2 else f"e{j}", B_NUMS[j], 2.5) for j in range(5)]
    cf = ctx("F", B_NUMS)
    rows += [row(cf, f"f{j}", B_NUMS[j], 2.5, gold="see note [6]" if j == 0 else None) for j in range(5)]
    cg = ctx("G", B_NUMS)
    rows += [row(cg, f"g{j}", 9001.5 + j, 9002.25) for j in range(5)]
    return rows


def fid(context: str) -> str:
    return Q.filing_id(context)


# the program evaluator and the usable rule

def test_program_value_reads_the_answer_without_exec():
    assert Q.program_value("  shares_balance = 6474102\n price = 9.10\n value = shares_balance * price\n"
                           " answer = value / 1000000") == pytest.approx(58.9143282)
    assert Q.program_value("a = 5\nb = -a\nanswer = (a - b) * 2") == 20
    assert Q.program_value("table_row = [829, 480, 104] # row labeled provision\nm = max(table_row)\nanswer = m") == 829
    assert Q.program_value("r = [1, 2, 3]\nanswer = sum(r) / len(r)") == 2
    assert Q.program_value("x = 1.1\nanswer = x ** 2") == pytest.approx(1.21)


@pytest.mark.parametrize("program", [
    "", "   \n  ", "import os\nanswer = 1", "answer = __import__('os')", "answer = open('x')",
    "answer = exec('1')", "import os\nanswer = os.getpid()", "a = 1\nanswer = a.real", "answer = max([1, 2], key=abs)",
    "answer = b + 1", "a = 5", "answer = 1 / 0", "answer = 9 ** 999", "answer = 'x'", "answer = True",
    "answer = [1, 2]", "def f(): pass\nanswer = 1", "answer = 2 if 1 else 3", "answer = sum([1] * 3)",
    "answer = len([1] + [2])",
])
def test_program_value_rejects_anything_outside_the_grammar(program):
    assert Q.program_value(program) is None


def test_usable_drops_an_unscorable_gold_and_a_program_that_does_not_reproduce():
    good = {"Question": "q", "Answer": "58.9", "Program": "a = 6474102\nb = 9.10\nanswer = a * b / 1000000"}
    assert Q.usable(good)
    # a percent gold against the program's fraction, as DFQ-1 reads it
    assert Q.usable({"Question": "q", "Answer": "4.4%", "Program": "a = 436\nb = 10000\nanswer = a / b"})
    assert not Q.usable({**good, "Answer": "[6] : shares were issued"})
    assert not Q.usable({**good, "Answer": "59.9"})
    assert not Q.usable({**good, "Program": ""})


# the filings

def test_filings_group_by_context_hash():
    rows = fixture()
    groups = Q.filings(rows)
    assert len(groups) == 7 and groups[fid(rows[5]["Context"])] == list(range(5, 11))
    assert all(len(k) == 8 for k in groups)


# write_task's extra files

def plain_write_task(tdir: Path, *, request: str, scorer: str, fill_repo) -> Path:
    """write_task with no extra files, written out here to check the default against"""
    tdir.mkdir(parents=True)
    labels, gold = fill_repo(tdir / "repo")
    (tdir / "task.toml").write_text(C.dump_toml({"request": request, "labels": labels}))
    (tdir / "gold.json").write_text(json.dumps(gold, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    shutil.copyfile(C.HERE / "check.py", tdir / "check.py")
    shutil.copyfile(C.HERE / scorer, tdir / "bench_score.py")
    return tdir


def test_write_task_default_adds_no_files(tmp_path):
    def fill(dst):
        dst.mkdir(parents=True)
        (dst / "filing.md").write_text("numbers\n")
        return {"benchmark": "docfinqa"}, {"benchmark": "docfinqa", "answer": "1"}
    new = C.write_task(tmp_path / "new", request="r", scorer="score_docfinqa.py", fill_repo=fill)
    old = plain_write_task(tmp_path / "old", request="r", scorer="score_docfinqa.py", fill_repo=fill)
    assert C.task_hash(new) == C.task_hash(old)
    more = C.write_task(tmp_path / "more", request="r", scorer="score_docfinqa.py", fill_repo=fill,
                        extra=("score_docfinqa.py",))
    assert (more / "score_docfinqa.py").exists() and C.task_hash(more) != C.task_hash(old)


# build.py's list writer and links

def test_write_list_and_link_use_the_test_dir(tmp_path, monkeypatch):
    monkeypatch.setitem(B.LIST_DIR, "full", tmp_path / "main")
    B.write_list("full", [{"id": "x", "benchmark": "docfinqa"}], tmp_path, selection={"seed": 1})
    doc = json.loads((tmp_path / "main" / "tasks.json").read_text())
    assert doc["selection"] == {"seed": 1} and doc["tasks"] == [{"id": "x", "benchmark": "docfinqa"}]
    (tmp_path / "docfinqa" / "x").mkdir(parents=True)
    B.link([{"id": "x", "benchmark": "docfinqa"}], tmp_path, clear=True, test_dir=tmp_path / "main")
    assert (tmp_path / "main" / "tasks" / "x").resolve() == (tmp_path / "docfinqa" / "x").resolve()
