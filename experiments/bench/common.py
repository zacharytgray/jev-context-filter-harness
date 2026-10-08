"""shared parts of the benchmark adapters: where the data lives, the task dir writer, the task hash, and the
offline facts of a task. stdlib only"""
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_DATA = ROOT / "data"  # $S1H_BENCH_DATA when unset; task dirs go to data/tasks
# no 0/O or 1/I, so a token reads back the same
TOKEN_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
HASH_SKIP_DIRS = {"__pycache__"}


def data_root() -> Path:
    return Path(os.environ.get("S1H_BENCH_DATA") or DEFAULT_DATA)


def tasks_root() -> Path:
    return data_root() / "tasks"


# toml writer: tomllib only reads. strings, ints, floats, bools, lists, and tables. None values are left out

def _tstr(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif o < 0x20 or o == 0x7F:
            out.append(f"\\u{o:04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _tkey(k: str) -> str:
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else _tstr(k)


def _tval(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return _tstr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_tval(x) for x in v if x is not None) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_tkey(k)} = {_tval(x)}" for k, x in v.items() if x is not None) + "}"
    raise TypeError(f"no toml form for {type(v).__name__}")


def dump_toml(doc: dict, _name: str = "") -> str:
    lines, tables = [], []
    for k, v in doc.items():
        if v is None:
            continue
        if isinstance(v, dict):
            tables.append((k, v))
        else:
            lines.append(f"{_tkey(k)} = {_tval(v)}")
    out = "\n".join(lines)
    for k, v in tables:
        name = f"{_name}.{_tkey(k)}" if _name else _tkey(k)
        out += f"\n\n[{name}]\n" + dump_toml(v, name)
    return out.strip("\n") + ("\n" if not _name else "")


def copy_repo(src: Path, dst: Path, skip: tuple = (".git",)) -> list[str]:
    """copy a tree without .git. a symlink to a file inside the tree becomes a copy of that file; any other symlink
    is left out. returns the left-out paths"""
    src = src.resolve()
    dropped = []
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d not in skip)
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for d in list(dirs):
            if (Path(root) / d).is_symlink():
                dirs.remove(d)
                dropped.append((rel / d).as_posix())
        for f in sorted(files):
            p = Path(root) / f
            if p.is_symlink():
                t = p.resolve()
                if t.is_file() and t.is_relative_to(src):
                    shutil.copyfile(t, dst / rel / f)
                else:
                    dropped.append((rel / f).as_posix())
                continue
            shutil.copyfile(p, dst / rel / f)
    return dropped


def write_task(tdir: Path, *, request: str, scorer: str, fill_repo, extra: tuple = ()) -> Path:
    """one task dir: task.toml, repo/, check.py, plus gold.json and the scorer as bench_score.py. fill_repo(repo
    path) writes the workdir snapshot and returns (labels, gold). extra: more files from experiments/bench copied in
    under their own names (a scorer's helper); none by default"""
    if tdir.exists():
        shutil.rmtree(tdir)
    tdir.mkdir(parents=True)
    labels, gold = fill_repo(tdir / "repo")
    (tdir / "task.toml").write_text(dump_toml({"request": request, "labels": labels}))
    (tdir / "gold.json").write_text(json.dumps(gold, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    shutil.copyfile(HERE / "check.py", tdir / "check.py")
    shutil.copyfile(HERE / scorer, tdir / "bench_score.py")
    for name in extra:
        shutil.copyfile(HERE / name, tdir / name)
    return tdir


def task_hash(tdir: Path) -> str:
    """sha256 over every file in the task dir: sorted relative paths, each with the sha256 of its bytes.
    __pycache__ dirs and .pyc files are left out"""
    items = []
    for root, dirs, files in os.walk(tdir):
        dirs[:] = [d for d in dirs if d not in HASH_SKIP_DIRS]
        for f in files:
            if f.endswith(".pyc"):
                continue
            p = Path(root) / f
            items.append((p.relative_to(tdir).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()))
    h = hashlib.sha256()
    for rel, digest in sorted(items):
        h.update(rel.encode() + b"\0" + digest.encode() + b"\n")
    return h.hexdigest()


def is_binary(b: bytes) -> bool:
    return b"\0" in b[:8192]


def text_of(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


def dir_facts(repo: Path) -> dict:
    """files, text chars (utf-8, bad bytes replaced), binary files and bytes"""
    n = text_chars = bins = bin_bytes = 0
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            b = (Path(root) / f).read_bytes()
            n += 1
            if is_binary(b):
                bins += 1
                bin_bytes += len(b)
            else:
                text_chars += len(text_of(b))
    return {"files": n, "text_chars": text_chars, "binary_files": bins, "binary_bytes": bin_bytes}


def token(code: str, key: str) -> str:
    """FIXTURE_<code>_<4 chars>, fixed by key so a rebuild gives the same token"""
    d = hashlib.sha256(key.encode()).digest()
    return f"FIXTURE_{code}_" + "".join(TOKEN_CHARS[b % len(TOKEN_CHARS)] for b in d[:4])


def marker(code: str, slug: str, key: str) -> dict:
    return {"path": f"receipts/{slug}.txt", "token": token(code, key)}
