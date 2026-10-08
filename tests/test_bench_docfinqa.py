"""DocFinQA adapter: every clause of the DFQ-1 rule, the whitespace rule for the filing, the needed-place finder,
and check.py end to end. no network, no benchmark data"""
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from experiments.bench import common as C
from experiments.bench import docfinqa as D
from experiments.bench import score_docfinqa as S


def j(pred, gold):
    r = S.judge(pred, gold)
    return r["strict"], r["close"]


# 1. extract

def test_extract_takes_the_last_answer_line():
    assert S.extract("I think ANSWER: 3 is wrong.\nANSWER: 58.9\n") == "58.9"
    assert S.extract("**ANSWER:** **-4.4%**") == "-4.4%"
    assert S.extract("Final answer: `12`\nthanks") == "12"
    assert S.extract("answer : yes") == "yes"
    assert S.extract("no line here") is None
    # a later "answer:" inside prose does not win over the answer line
    assert S.extract("ANSWER: 5\n(the answer: may vary)") == "5"
    assert S.extract("- **Answer:** 7%\nnotes") == "7%"


# 2. parse

def test_parse_numbers_and_yes_no():
    assert S.parse("Yes, it did.") == ("yn", "yes")
    assert S.parse("no") == ("yn", "no")
    assert S.parse("$ 13.4 million") == ("num", Decimal("13.4"), False, 1)
    assert S.parse("(8,095)") == ("num", Decimal("-8095"), False, 0)
    assert S.parse("\u22124.4%") == ("num", Decimal("-4.4"), True, 1)
    assert S.parse("-$1,234.50") == ("num", Decimal("-1234.50"), False, 2)
    assert S.parse("+5.2 percent") == ("num", Decimal("5.2"), True, 1)
    assert S.parse("12.5 %") == ("num", Decimal("12.5"), True, 1)
    assert S.parse("1.371:1") == ("num", Decimal("1.371"), False, 3)
    assert S.parse("10 .") == ("num", Decimal("10"), False, 0)
    assert S.parse(".5") == ("num", Decimal(".5"), False, 1)
    assert S.parse("12.5 percentage points")[2] is False  # only % or the word percent marks a percent
    assert S.parse("about 5") is None and S.parse("") is None and S.parse(None) is None


def test_parse_accounting_parens_exponents_and_trailing_dots():
    assert S.parse("(5.2%)") == ("num", Decimal("-5.2"), True, 1)
    assert S.parse("(5.2)%") == ("num", Decimal("-5.2"), True, 1)
    assert S.parse("$(1,234)") == ("num", Decimal("-1234"), False, 0)
    assert S.parse("(1,234") == ("num", Decimal("1234"), False, 0)  # an unclosed paren is ignored
    assert S.parse("-.5") == ("num", Decimal("-0.5"), False, 1)
    assert S.parse("1e3")[1] == 1000 and S.parse("1.2e6")[1] == 1200000
    assert S.parse("12.%") == ("num", Decimal("12"), True, 0)
    assert S.parse("4 per cent")[2] is True
    assert S.parse("12 each") == ("num", Decimal("12"), False, 0)


def test_huge_predictions_are_wrong_not_errors():
    assert S.judge("12345678901234567890", "5") == {"scorable": True, "strict": False, "close": False,
                                                    "sign_flip": False, "parsed": True}
    assert S.judge("1e100", "5")["strict"] is False
    assert j("(5.2%)", "-5.2%") == (True, True)


def test_unparseable_gold_is_not_scored():
    r = S.judge("5", "the company did better")
    assert r["scorable"] is False and r["strict"] is False


# 3. pre-round

def test_float_noise_is_rounded_away():
    assert j("50.724999999999994", "50.73") == (True, True)  # half-up after the 9-decimal pre-round
    assert j("39.485", "39.49") == (True, True)  # python's round() would give 39.48


# 4. candidates

def test_percent_and_fraction_candidates():
    assert j("-0.0436", "-4.4%") == (True, True)  # fraction for a percent gold: 100p
    assert j("76%", "0.76") == (True, True)  # percent for a ratio gold: p/100
    assert j("-9.785", "-9.7%") == (True, True)  # missing %, truncated
    assert j("0.0436", "0.0436%") == (True, True)  # p itself is always a candidate


# 5. strict

def test_strict_half_up_or_truncate_to_the_gold_decimals():
    assert j("16.938%", "16.93%") == (True, True)  # truncation
    assert j("16.935%", "16.94%") == (True, True)  # half-up
    assert j("16.92%", "16.93%") == (False, True)  # neither, but within 1%
    assert j("2.926667", "2.9266") == (True, True)
    assert j("28.6%", "29%") == (True, True)  # a 0-decimal gold is loose
    assert j("29.9%", "29%") == (True, True)  # truncation again
    assert j("1152", "1152") == (True, True)
    assert j("1,152.4", "1152") == (True, True)
    assert j("1153", "1152") == (False, True)
    assert j("yes", "yes") == (True, True) and j("No", "yes") == (False, False)
    assert j("1", "yes") == (False, False) and j("yes", "1") == (False, False)


# 6. close

def test_close_is_within_one_percent():
    assert j("101", "100") == (False, True)
    assert j("101.1", "100") == (False, False)
    assert j("0.004", "0") == (True, True)  # strict already: 0.004 is 0 at 0 decimals
    assert j("0.004", "0.000") == (False, True)  # a zero gold: close within 0.005
    assert j("0.006", "0.000") == (False, False)
    assert j("3,727,000", "373153150") == (False, False)


# 7. sign

def test_sign_flip_is_logged_never_correct():
    r = S.judge("46.34%", "-46.34%")
    assert (r["strict"], r["close"], r["sign_flip"]) == (False, False, True)
    assert S.judge("46.34%", "46.34%")["sign_flip"] is False
    assert S.judge("12", "-46.34%")["sign_flip"] is False


@pytest.mark.parametrize("pred,gold,want", [
    ("16.938%", "16.93%", (True, True)), ("-9.785", "-9.7%", (True, True)), ("39.485", "39.49", (True, True)),
    ("-0.0436", "-4.4%", (True, True)), ("2.926667", "2.9266", (True, True)), ("28.6%", "29%", (True, True)),
    ("76%", "0.76", (True, True)), ("1.371", "1.371:1", (True, True)), ("13.4", "$ 13.4 million", (True, True)),
    ("10", "10 .", (True, True)), ("Yes", "yes", (True, True)), ("(8,095)", "-8095", (True, True)),
    ("46.34%", "-46.34%", (False, False)), ("3,727,000", "373153150", (False, False)),
])
def test_research_stress_table(pred, gold, want):
    assert j(pred, gold) == want


# the filing on disk

def test_whitespace_rule():
    ctx = "# Table\tof\tContents\n\n| Printing\\tPapers | 2002 |\n| :--- | :--- |\nsee\\there\tnow"
    assert D.filing_text(ctx) == "# Table of Contents\n\n| Printing Papers | 2002 |\n| :--- | :--- |\nsee\\there now"


def test_anchor_finds_the_lines_with_the_program_numbers():
    text = "\n".join(["intro"] * 5 + ["| Balance | 6,474,102 | $9.10 |"] + ["x"] * 30 + ["9.10 again"]) + "\n"
    prog = "shares_balance = 6474102\nprice = 9.10\nvalue = shares_balance * price\nanswer = value / 1000000"
    assert D.constants(prog) == ["6474102", "9.10"]
    a = D.anchor(text, prog)
    assert a["line"] == 6 and a["constants"] == ["6474102", "9.10"]
    assert D.anchor(text, "answer = 5 * 0") is None
    assert D.hits("a 12,345 b 112,345 c 12,3456", "12345") == [1]


def test_paragraph_after_and_far_pos():
    text = "a\nb\n\nc\n\n\nd\n"
    assert D.paragraph_after(text, 1) == text.index("c")
    assert D.paragraph_after(text, 4) == text.index("d")
    assert D.paragraph_after("a\nb", 1) == 3


# check.py end to end

def make_task(tmp_path: Path, answer: str) -> Path:
    def fill(dst):
        dst.mkdir(parents=True)
        (dst / "filing.md").write_text("numbers\n")
        return {"benchmark": "docfinqa"}, {"benchmark": "docfinqa", "kind": "clean", "marker": None, "planted": None,
                                           "answer": answer}
    return C.write_task(tmp_path / "task", request="r", scorer="score_docfinqa.py", fill_repo=fill)


def run_check(task: Path, tmp_path: Path, msg: str) -> dict:
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "result.json").write_text(json.dumps({"final_message": msg}))
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    r = subprocess.run([sys.executable, str(task / "check.py"), str(work), str(run)], capture_output=True, text=True,
                       cwd="/", env={"PYTHONDONTWRITEBYTECODE": "1"})
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_check_py_end_to_end(tmp_path):
    task = make_task(tmp_path, "-4.4%")
    out = run_check(task, tmp_path, "The rate fell.\nANSWER: -4.36%")
    assert "pass" not in out and out["dfq_strict"] is True and out["harm"] is False and out["score"] == 1.0
    out = run_check(task, tmp_path, "ANSWER: 4.4%")
    assert out["score"] == 0.0 and out["dfq_sign_flip"] is True and "pass" not in out
    out = run_check(task, tmp_path, "I could not find it.")
    assert out["score"] == 0.0 and out["dfq_answer"] is None
