"""docs/PROMPTS.md against the code that sends the prompts, and against recorded runs when the data is unpacked"""
import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

from experiments.bench import common as BC
from s1harness import messages as M
from s1harness import tools as T

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dump_prompts", ROOT / "scripts/dump_prompts.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
DOC = (ROOT / "docs/PROMPTS.md").read_text()
# every shipped run set
RUN_SETS = tuple(sorted(p.name for p in (ROOT / "experiments").glob("runs-*-*k")))


def test_doc_matches_code():
    # everything but the example requests, which need the task data
    assert D.strip_data(DOC) == D.render(with_data=False), "run scripts/dump_prompts.py"


def test_doc_matches_code_and_task_data():
    if D.examples() is None:
        pytest.skip("task data not unpacked (scripts/unpack_data.py)")
    assert DOC == D.render(), "run scripts/dump_prompts.py"


def first_runs() -> list[Path]:
    """the first recorded run of each configuration in each run set, if unpacked"""
    out = []
    for rs in RUN_SETS:
        for records in sorted((ROOT / "experiments" / rs / "main").glob("*/records.jsonl")):
            run = ROOT / json.loads(records.read_text().splitlines()[0])["run_dir"]
            if (run / "transcript.json").exists():
                out.append(run)
    return out


def test_recorded_runs_match_doc():
    runs = first_runs()
    if not runs:
        pytest.skip("run data not unpacked (scripts/unpack_data.py)")
    for run in runs:
        levers = json.loads((run / "levers.json").read_text())
        transcript = json.loads((run / "transcript.json").read_text())
        assert transcript[0] == {"role": "system", "content": M.system_prompt()}, run
        assert json.loads((run / "tools.json").read_text()) == T.schemas(levers["relevance"]), run
        request = (run.parents[1] / "request.txt").read_text()
        assert transcript[1] == {"role": "user", "content": request}, run
        task = run.parents[2].name
        toml = next(BC.tasks_root().glob(f"*/{task}/task.toml"), None)
        if toml is not None:
            # the cli reads request.txt in text mode, so \r\n and \r arrive as \n
            sent = tomllib.loads(toml.read_text())["request"].replace("\r\n", "\n").replace("\r", "\n")
            assert sent == request, run
