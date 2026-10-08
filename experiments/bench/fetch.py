"""download the public data the benchmark adapters read, into $S1H_BENCH_DATA (default: the repo's data/). no login
is needed for any of it.

    PYTHONPATH=$PWD python experiments/bench/fetch.py [--official]

- contextbench/data/{contextbench_verified,full}.parquet from huggingface (Contextbench/ContextBench at a pinned
  revision), converted to .jsonl with pyarrow in a throwaway uv env (the harness venv has no pyarrow)
- contextbench/repo: the ContextBench code at 1436c28, only with --official (about 300 MB; needed for the official
  evaluator, see cb_official.py). repo checkouts per task are fetched by contextbench.checkout when a task is built
- legalbenchrag/LegalBench-RAG.zip from the dropbox link in the LegalBench-RAG README, unzipped to data/
- docfinqa/test.json from huggingface (kensho/DocFinQA at a pinned revision), about 580 MB
each file is checked against its sha256 when one is known. existing files are kept."""
import argparse
import hashlib
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.bench.common import data_root  # noqa: E402

CB_REV = "c2855792b006af41c67202d33883fb9d46362853"
CB_CODE = "https://github.com/EuniAI/ContextBench"
CB_CODE_COMMIT = "1436c28a8eb95496da4ea69ad458b9f8a8eb7d61"
DFQ_REV = "64ebaff62f692495bcc182f45cf9a9606251b19b"
LB_ZIP = ("https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4"
          "?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=1")
FILES = [
    ("contextbench/data/contextbench_verified.parquet",
     f"https://huggingface.co/datasets/Contextbench/ContextBench/resolve/{CB_REV}/data/contextbench_verified.parquet",
     "e9dcfd504cbfb849ac815a79040c793d0d92f94eecc9b5a4ee3e1445a2f8a791"),
    ("contextbench/data/full.parquet",
     f"https://huggingface.co/datasets/Contextbench/ContextBench/resolve/{CB_REV}/data/full.parquet",
     "2f56535bdc73eb8a68bf4ebb49789d8e9cd4f219ea60df6290b85278aee61ca8"),
    ("legalbenchrag/LegalBench-RAG.zip", LB_ZIP,
     "27431be37db9b1db23f8ab790a42d076adb1f72d7f9e7562e36a10573405f88d"),
    ("docfinqa/test.json", f"https://huggingface.co/datasets/kensho/DocFinQA/resolve/{DFQ_REV}/test.json",
     "0ebf95ccc719e72f45333e256928c90e8fa7a9a81f226dcb38ad7b3a48a3ff42"),
]
TO_JSONL = """
import json, sys, pyarrow.parquet as pq
for src in sys.argv[1:]:
    with open(src[:-len('.parquet')] + '.jsonl', 'w') as f:
        for r in pq.read_table(src).to_pylist():
            f.write(json.dumps(r) + '\\n')
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get(rel: str, url: str, digest: str):
    p = data_root() / rel
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {rel}", flush=True)
        tmp = p.with_suffix(p.suffix + ".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while b := r.read(1 << 20):
                f.write(b)
        tmp.rename(p)
    if digest and sha256(p) != digest:
        raise SystemExit(f"{rel}: sha256 differs from {digest}")
    print(f"ok {rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", action="store_true", help="also clone the ContextBench code for its evaluator")
    a = ap.parse_args()
    for rel, url, digest in FILES:
        get(rel, url, digest)
    d = data_root() / "contextbench" / "data"
    todo = [str(d / f"{n}.parquet") for n in ("contextbench_verified", "full") if not (d / f"{n}.jsonl").exists()]
    if todo:
        subprocess.run(["uv", "run", "--no-project", "--with", "pyarrow", "python", "-c", TO_JSONL, *todo],
                       check=True)
    lb = data_root() / "legalbenchrag"
    if not (lb / "data" / "benchmarks").is_dir():
        with zipfile.ZipFile(lb / "LegalBench-RAG.zip") as z:
            z.extractall(lb / "data")
    if a.official:
        repo = data_root() / "contextbench" / "repo"
        if not (repo / ".git").is_dir():
            subprocess.run(["git", "clone", "-q", CB_CODE, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", CB_CODE_COMMIT], check=True)
    print(f"data in {data_root()}")


if __name__ == "__main__":
    main()
