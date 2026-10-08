"""optional, offline: score ContextBench runs with the official evaluator (github.com/EuniAI/ContextBench at 1436c28),
for the block (symbol) level our scorer does not compute, and to check our file and line numbers against it.

    PYTHONPATH=$PWD python experiments/bench/cb_official.py export <task_dir>:<run_dir> ... > pred.jsonl
    PYTHONPATH=$PWD python experiments/bench/cb_official.py run pred.jsonl out.jsonl
    PYTHONPATH=$PWD python experiments/bench/cb_official.py selftest

needs `fetch.py --official` (the code in $S1H_BENCH_DATA/contextbench/repo) and uv. the evaluator runs in its own
throwaway env: python 3.11 with tree-sitter 0.20.4 and tree-sitter-languages 1.10.2 (no wheels past 3.11) and pyarrow.
it checks out each repo as a git worktree of our local checkout, under $S1H_BENCH_DATA/contextbench/evalcache.
export maps a planted task's declared lines back to the original file (planted lines dropped), so planted runs can be
scored too. pred_steps carries the reached context, one step per tool call that reached file lines, so the official
trajectory metrics (AUC, redundancy) come out as well. export parses the final message with our parser, so selftest
checks the parser on its own: it runs the official extract.py functions (_extract_tag_blocks, _parse_file_lines_pairs)
on every test message and compares. then it scores hand-made final messages both ways and fails on any difference in
file or line recall, precision, hit or size."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.bench import common as C  # noqa: E402
from experiments.bench import contextbench as CB  # noqa: E402
from experiments.bench import score_contextbench as S  # noqa: E402

UV = ["uv", "run", "--no-project", "--python", "3.11", "--with", "tree-sitter==0.20.4", "--with",
      "tree-sitter-languages==1.10.2", "--with", "pyarrow"]


def unplant(spans: dict, planted: dict | None) -> dict:
    """{file: [(a, b)]} in the planted file's lines -> the original file's lines"""
    if not planted:
        return spans
    f, first, last = planted["file"], planted["first_line"], planted["last_line"]
    k = last - first + 1
    out = {}
    for p, iv in spans.items():
        if not (p == f or p.endswith("/" + f) or f.endswith("/" + p.lstrip("./"))):
            out[p] = iv
            continue
        for a, b in iv:
            if a < first:
                out.setdefault(p, []).append((a, min(b, first - 1)))
            if b > last:
                out.setdefault(p, []).append((max(a, last + 1) - k, b - k))
    return out


def export_one(task_dir: Path, run_dir: Path) -> dict:
    labels = tomllib.loads((task_dir / "task.toml").read_text())["labels"]
    spec = json.loads((task_dir / "gold.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    spans = unplant(S.parse_patch_context(result.get("final_message") or "") or {}, spec.get("planted"))
    steps = []
    tr = run_dir / "transcript.json"
    if tr.exists():
        files = S.Files(str(task_dir / "repo"))
        calls = {}
        for m in json.loads(tr.read_text()):
            for c in m.get("tool_calls") or []:
                fn = c.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls[c.get("id", "")] = (fn.get("name", ""), args if isinstance(args, dict) else {})
            if m.get("role") != "tool":
                continue
            name, args = calls.get(m.get("tool_call_id", ""), ("", {}))
            content = m.get("content") or ""
            if name == "read":
                g, _, _ = S.read_reached(str(args.get("path") or ""), content, files)
            elif name == "bash":
                g, _, _ = S.bash_reached(str(args.get("command") or ""), content, files)
            else:
                continue
            g = unplant({f: S.as_intervals(v) for f, v in g.items() if v}, spec.get("planted"))
            if g:
                steps.append({"files": sorted(g), "spans": {f: [{"start": a, "end": b} for a, b in v]
                                                            for f, v in g.items()}})
    return {"instance_id": labels["instance_id"], "repo_url": str(CB.root() / "checkouts" / labels["item"]),
            "commit": labels["base_commit"],
            "traj_data": {"pred_steps": steps, "pred_files": sorted(spans),
                          "pred_spans": {f: [{"start": a, "end": b} for a, b in v] for f, v in spans.items()}},
            "model_patch": ""}


def run(pred: Path, out: Path) -> None:
    repo = CB.root() / "repo"
    cache = CB.root() / "evalcache"
    env = {**os.environ, "CONTEXTBENCH_TMP_ROOT": str(cache / "tmp")}
    subprocess.run(UV + ["python", "-m", "contextbench.evaluate", "--gold",
                         str(CB.root() / "data" / f"{CB.SUBSET}.parquet"), "--pred", str(pred.resolve()),
                         "--cache", str(cache / "repos"), "--out", str(out.resolve())],
                   cwd=repo, env=env, check=True, stderr=subprocess.DEVNULL)


# hand-made final messages: (item, message). each tests one parsing or scoring rule
CASES = [
    ("psf__requests-1142", "<PATCH_CONTEXT>\nFile: requests/models.py\nLines: 332-387\n\nFile: requests/models.py\n"
                           "Lines: 386-398\n</PATCH_CONTEXT>"),
    ("psf__requests-1142", "done.\n<PATCH_CONTEXT>\nFile: requests/models.py\nLines: 332-342\n\nFile: requests/api.py\n"
                           "Lines: 1-40\n</PATCH_CONTEXT>"),
    ("psf__requests-1142", "<PATCH_CONTEXT>\nFile: /testbed/requests/models.py\nLines: 398-380\nFile: requests/nope.py\n"
                           "Lines: 1-10\nFile: ./requests/sessions.py\nLines: 1-5\nFile: requests/hooks.py\n"
                           "Lines: 12\n</PATCH_CONTEXT>"),
    ("psf__requests-1142", "<PATCH_CONTEXT>\nFile: requests/models.py\nLines: 332-340\nLines: 341-350\nLines: 700-900\n"
                           "</PATCH_CONTEXT>"),
    ("psf__requests-1142", "first try\n<PATCH_CONTEXT>\nFile: requests/api.py\nLines: 1-10\n</PATCH_CONTEXT>\nthen\n"
                           "<PATCH_CONTEXT>\nFile: /private/tmp/s1h-eval/x/requests/models.py\nLines: 300 - 420\n"
                           "</PATCH_CONTEXT>"),
    ("iamkun__dayjs-1611", "<PATCH_CONTEXT>\nFile: src/plugin/duration/index.js\nLines: 80-95\n\nFile: src/index.js\n"
                           "Lines: 1-20\n</PATCH_CONTEXT>"),
    ("django__django-10880", "<PATCH_CONTEXT>\nFile: django/db/models/aggregates.py\nLines: 1-120\n</PATCH_CONTEXT>"),
    ("pytest-dev__pytest-7432", "<PATCH_CONTEXT>\nFile: src/_pytest/skipping.py\nLines: 225-306\nFile: "
                                "src/_pytest/outcomes.py\nLines: 1-30\n</PATCH_CONTEXT>"),
]


# parse-only cases, on top of CASES
PARSE_ONLY = [
    "no block at all",
    "<PATCH_CONTEXT>\nFile: a.py\nLines: 1-2\n",
    "<PATCH_CONTEXT>\r\nFile: a.py\r\nLines: 3-4\r\n</PATCH_CONTEXT>",
    "<patch_context>\nFile: a.py\nLines: 1-2\n</patch_context>",
    "<PATCH_CONTEXT>File: a.py Lines: 1-2</PATCH_CONTEXT>",
    "<PATCH_CONTEXT>\n  File:   /home/repo/x/y.py  \n  Lines:  9 -  7 extra\nLines: a-b\nFile:\nLines: 1-1\n</PATCH_CONTEXT>",
    "<PATCH_CONTEXT>\nFile: 'q.py'\nLines: 1-2\nFile: \"/testbed/r.py\"\nLines: 5-6\n</PATCH_CONTEXT>",
]
OFFICIAL_PARSE = """
import json, sys
from contextbench.agents.minisweagent.extract import _extract_tag_blocks, _parse_file_lines_pairs
out = []
for msg in json.load(sys.stdin):
    blocks = _extract_tag_blocks(msg, tag="PATCH_CONTEXT")
    out.append({f: [[s["start"], s["end"]] for s in v] for f, v in _parse_file_lines_pairs(blocks[-1]).items()}
               if blocks else None)
print(json.dumps(out))
"""


def parse_check() -> int:
    """our parse_patch_context against the official extract.py functions, on every test message"""
    msgs = [m for _, m in CASES] + PARSE_ONLY
    r = subprocess.run(UV + ["python", "-c", OFFICIAL_PARSE], cwd=CB.root() / "repo", input=json.dumps(msgs),
                       capture_output=True, text=True, check=True)
    official = json.loads(r.stdout.strip().splitlines()[-1])
    bad = 0
    for m, o in zip(msgs, official):
        ours = S.parse_patch_context(m)
        ours = {f: [list(x) for x in v] for f, v in ours.items()} if ours is not None else None
        if ours != o:
            bad += 1
            print(f"parse DIFF on {m!r}: official {o} ours {ours}")
    print(f"parser: {len(msgs) - bad} of {len(msgs)} messages parse the same as extract.py")
    return bad


def selftest() -> int:
    rows = CB.load()
    tmp = Path(tempfile.mkdtemp(prefix="cb-official-", dir=C.data_root()))
    built, pairs = {}, []
    for k, (item, msg) in enumerate(CASES):
        if item not in built:
            built[item] = CB.build(rows[item], {"kind": "clean"}, tmp / "tasks")
        rd = tmp / "runs" / f"r{k}"
        rd.mkdir(parents=True)
        (rd / "result.json").write_text(json.dumps({"final_message": msg}))
        pairs.append((built[item], rd))
    pred = tmp / "pred.jsonl"
    pred.write_text("".join(json.dumps(export_one(t, r)) + "\n" for t, r in pairs))
    out = tmp / "out.jsonl"
    run(pred, out)
    official = [json.loads(x) for x in out.read_text().splitlines()]
    bad = 0
    for (t, r), o in zip(pairs, official):
        spec = json.loads((t / "gold.json").read_text())
        ours = S.score(t, t / "repo", r, spec)
        if "error" in o:
            same = not ours["cb_decl_found"] or ours["cb_decl_file_pred"] == 0
            print(f"{r.name} {t.name}: official error {o['error']}, ours zero: {same}")
            bad += not same
            continue
        diffs = []
        for lvl in ("file", "line"):
            for ok, mk in (("coverage", "recall"), ("precision", "precision"), ("intersection", "hit"),
                           ("gold_size", "gold"), ("pred_size", "pred")):
                a, b = o["final"][lvl][ok], ours[f"cb_decl_{lvl}_{mk}"]
                if abs(a - b) > 1e-12:
                    diffs.append(f"{lvl} {ok}: official {a} ours {b}")
        sym = o["final"]["symbol"]
        print(f"{r.name} {t.name}: file R/P {o['final']['file']['coverage']:.3f}/{o['final']['file']['precision']:.3f}"
              f" line R/P {o['final']['line']['coverage']:.3f}/{o['final']['line']['precision']:.3f}"
              f" block R/P {sym['coverage']:.3f}/{sym['precision']:.3f} "
              + ("same" if not diffs else "DIFF " + "; ".join(diffs)))
        bad += bool(diffs)
    print(f"{len(pairs) - bad} of {len(pairs)} agree")
    if bad:
        print(f"files kept in {tmp}")
    else:
        shutil.rmtree(tmp)
    return bad


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "export":
        for pair in sys.argv[2:]:
            t, r = pair.split(":", 1)
            print(json.dumps(export_one(Path(t), Path(r))))
    elif sys.argv[1] == "run":
        run(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == "selftest":
        raise SystemExit(1 if parse_check() + selftest() else 0)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
