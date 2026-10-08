#!/usr/bin/env python3
"""unpack the data tarballs in data/ into the repo, offline.

    python3 scripts/unpack_data.py [--only NAME[,NAME...]] [--no-verify]

checks every tarball against data/SHA256SUMS, unpacks the run folders in place (experiments/runs-*/) and the task dirs
to data/tasks/<benchmark>/<task id>/, checks each task dir against the sha256 in experiments/main/tasks.json, and links
each task into experiments/main/tasks/, where experiments/run.py looks for them.

python 3.14 reads zstd with its standard library (compression.zstd); older pythons need the zstd command on PATH."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TASKS = ROOT / "experiments" / "main" / "tasks.json"
LINKS = ROOT / "experiments" / "main" / "tasks"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def task_hash(tdir: Path) -> str:
    # experiments/bench/common.py task_hash: sorted (relative path, sha256 of bytes), pycache left out
    items = []
    for root, dirs, files in os.walk(tdir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(".pyc"):
                p = Path(root) / f
                items.append((p.relative_to(tdir).as_posix(), sha256_file(p)))
    h = hashlib.sha256()
    for rel, d in sorted(items):
        h.update(rel.encode() + b"\0" + d.encode() + b"\n")
    return h.hexdigest()


def open_zst(p: Path):
    try:
        from compression import zstd  # python 3.14+
    except ImportError:
        if not shutil.which("zstd"):
            raise SystemExit("needs python 3.14+ or the zstd command on PATH")
        proc = subprocess.Popen(["zstd", "-d", "-q", "-c", "--long=31", str(p)], stdout=subprocess.PIPE)
        return proc.stdout, proc
    # the tarballs use a 1 GB window (zstd --long=30)
    opts = {zstd.DecompressionParameter.window_log_max: 31}
    return zstd.ZstdFile(p, options=opts), None


def unpack(p: Path):
    f, proc = open_zst(p)
    with f, tarfile.open(fileobj=f, mode="r|") as t:
        t.extractall(ROOT, filter="data")
    if proc and proc.wait():
        raise SystemExit(f"zstd failed on {p.name}")


def main():
    ap = argparse.ArgumentParser(description="unpack the data tarballs into the repo")
    ap.add_argument("--only", default=None, help="comma list of tarball names (without .tar.zst)")
    ap.add_argument("--no-verify", action="store_true", help="skip the sha256 checks")
    a = ap.parse_args()
    sums = {}
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name] = digest
    names = sorted(sums)
    if a.only:
        want = {n if n.endswith(".tar.zst") else n + ".tar.zst" for n in a.only.split(",")}
        missing = want - set(names)
        if missing:
            raise SystemExit(f"not in data/SHA256SUMS: {', '.join(sorted(missing))}")
        names = sorted(want)
    for name in names:
        p = DATA / name
        if not a.no_verify:
            got = sha256_file(p)
            if got != sums[name]:
                raise SystemExit(f"{name}: sha256 {got} is not the one in data/SHA256SUMS")
        print(f"unpacking {name}", flush=True)
        unpack(p)
    tasks = json.loads(TASKS.read_text())["tasks"]
    LINKS.mkdir(exist_ok=True)
    present = bad = 0
    for t in tasks:
        tdir = DATA / "tasks" / t["benchmark"] / t["id"]
        if not tdir.is_dir():
            continue
        present += 1
        if not a.no_verify and task_hash(tdir) != t["sha256"]:
            print(f"DIFF {t['id']}: does not hash as experiments/main/tasks.json says", flush=True)
            bad += 1
        link = LINKS / t["id"]
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(tdir, LINKS))
    print(f"{present} of {len(tasks)} task dirs present and linked in experiments/main/tasks/"
          + ("" if a.no_verify else f", {present - bad} hash as tasks.json says"))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
