"""render one run directory as a self-contained html page"""
import argparse
import html
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("view_template.html")


def _json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _jsonl(p: Path):
    if not p.exists():
        return None
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _record(run_dir: Path):
    # the experiment's graded record for this run (score, detail), from the nearest records.jsonl above it
    d = run_dir.resolve()
    for up in list(d.parents)[:6]:
        f = up / "records.jsonl"
        if not f.exists():
            continue
        rel = d.relative_to(up).as_posix()
        hit = None
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:  # blank, or a line a running sweep is still writing
                continue
            if isinstance(r, dict) and str(r.get("run_dir", "")).endswith("/" + rel):
                hit = r
        if hit:
            return hit
    return None


def load_run(run_dir: Path) -> dict:
    # raw files as they are. the page does the merging. missing files are null
    return {
        "id": run_dir.name,
        "result": _json(run_dir / "result.json"),
        "levers": _json(run_dir / "levers.json"),
        "tools": _json(run_dir / "tools.json"),
        "transcript": _json(run_dir / "transcript.json"),
        "llm": _jsonl(run_dir / "llm.jsonl"),
        "trace": _jsonl(run_dir / "trace.jsonl"),
        "record": _record(run_dir),
    }


def render(data: dict) -> str:
    # no sort_keys: criteria order is meaningful and file order is already stable
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # safe inside a script tag
    blob = blob.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    page = TEMPLATE.read_text()
    page = page.replace("__RUN_TITLE__", html.escape(f"run {data['id']}"), 1)
    return page.replace("__RUN_DATA__", blob, 1)


def main():
    ap = argparse.ArgumentParser(prog="s1h-view", description="write view.html for one run")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <run_dir>/view.html")
    a = ap.parse_args()
    out = a.out or a.run_dir / "view.html"
    out.write_text(render(load_run(a.run_dir)))
    print(out)


if __name__ == "__main__":
    main()
