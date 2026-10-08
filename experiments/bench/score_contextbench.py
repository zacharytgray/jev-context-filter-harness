"""ContextBench scorer (arXiv:2602.05892), stdlib only. copied into each task dir as bench_score.py.

declared context: the last <PATCH_CONTEXT> block of the final message, parsed and scored the way the official code
does it (github.com/EuniAI/ContextBench at 1436c28): agents/minisweagent/extract.py (block and File/Lines pairs),
evaluate.py:43-111 (paths resolved by suffix against the checkout, missing files dropped), metrics/compute.py
(coverage = recall, precision, merged inclusive line intervals, gold empty -> recall 1.0, pred empty -> precision 1.0).
F1 is not in the official code; it is 2PR / (P + R), 0 when both are 0.

reached context: the gold lines that reached the agent, from transcript.json alone (tool call args and tool message
text), so it works in every configuration. see reached()."""
import json
import os
import re
import shlex
from pathlib import Path

PATCH_RX = re.compile(r"<PATCH_CONTEXT>\s*([\s\S]*?)\s*</PATCH_CONTEXT>")


# paths

def norm_gold_path(p: str) -> str:
    """parsers/gold.py:_normalize_rel_path"""
    if not p:
        return ""
    p = p.replace("\\", "/")
    if p.startswith("/testbed/"):
        return p[len("/testbed/"):]
    if p.startswith("/workspace/"):
        parts = p[len("/workspace/"):].split("/", 1)
        return parts[1] if len(parts) == 2 else parts[0]
    if p.startswith("/"):
        return p.lstrip("/")
    return p.lstrip("./")


def norm_pred_path(p: str) -> str:
    """agents/minisweagent/extract.py:_normalize_path with no repo dir name"""
    p = (p or "").strip().strip("'\"")
    if not p:
        return ""
    if p.startswith("/testbed/"):
        p = p[len("/testbed/"):]
    if p.startswith("/home/"):
        parts = p[len("/home/"):].split("/", 1)
        if len(parts) != 2:
            return ""
        p = p[len("/home/"):]
    if p.startswith("/"):
        p = p.lstrip("/")
    if p.startswith("./"):
        p = p[2:]
    return p


def _is_repo_file(repo: str, rel: str) -> bool:
    """evaluate.py:_is_repo_file"""
    if not repo or not rel:
        return False
    p = str(rel).strip()
    if not p or os.path.isabs(p):
        return False
    real = os.path.realpath(repo)
    target = os.path.realpath(os.path.join(real, p))
    try:
        if os.path.commonpath([real, target]) != real:
            return False
    except ValueError:
        return False
    return os.path.isfile(target)


def resolve(repo: str, path: str) -> str:
    """evaluate.py:_resolve_repo_relpath: try the path, then drop leading parts until a file exists. "" if none"""
    if not repo or not path:
        return ""
    p = str(path).strip().strip("'\"").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    parts = [x for x in p.split("/") if x]
    for i in range(len(parts)):
        cand = "/".join(parts[i:])
        if _is_repo_file(repo, cand):
            return cand
    return ""


# intervals, metrics/compute.py

def merge(iv: list) -> list:
    """merge overlapping or adjacent inclusive intervals"""
    if not iv:
        return []
    s = sorted(iv)
    out = [s[0]]
    for a, b in s[1:]:
        la, lb = out[-1]
        if a <= lb + 1:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def total(by_file: dict) -> int:
    return sum(b - a + 1 for iv in by_file.values() for a, b in merge(iv))


def _inter(a: list, b: list) -> int:
    a, b = merge(a), merge(b)
    i = j = n = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo <= hi:
            n += hi - lo + 1
        if a[i][1] < b[j][1]:
            i += 1
        elif b[j][1] < a[i][1]:
            j += 1
        else:
            i += 1
            j += 1
    return n


def inter(a: dict, b: dict) -> int:
    return sum(_inter(a.get(f, []), b.get(f, [])) for f in set(a) | set(b))


def cov_prec(pred: int, gold: int, hit: int) -> tuple[float, float]:
    """compute.py:coverage_precision"""
    return (hit / gold if gold > 0 else 1.0), (hit / pred if pred > 0 else 1.0)


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def gold_sets(gold: list[dict]) -> tuple[set, dict]:
    """gold files and merged line intervals. gold paths are normalized already; they are kept even when the file is
    not in the checkout, as the official Gold.files() keeps them"""
    files, lines = set(), {}
    for g in gold:
        f = norm_gold_path(g["file"])
        if not f:
            continue
        files.add(f)
        lines.setdefault(f, []).append((int(g["start_line"]), int(g["end_line"])))
    return files, {f: merge(v) for f, v in lines.items()}


def level_scores(pred_files: set, pred_lines: dict, gold_files: set, gold_lines: dict) -> dict:
    fh = len(pred_files & gold_files)
    fr, fp = cov_prec(len(pred_files), len(gold_files), fh)
    lp, lg, lh = total(pred_lines), total(gold_lines), inter(pred_lines, gold_lines)
    lr, lpr = cov_prec(lp, lg, lh)
    return {"file_recall": fr, "file_precision": fp, "file_f1": f1(fp, fr), "file_hit": fh,
            "file_pred": len(pred_files), "file_gold": len(gold_files),
            "line_recall": lr, "line_precision": lpr, "line_f1": f1(lpr, lr), "line_hit": lh,
            "line_pred": lp, "line_gold": lg}


# declared context

def parse_patch_context(message: str) -> dict | None:
    """the last <PATCH_CONTEXT> block as {path: [(a, b)]} with paths normalized but not resolved. None when there is
    no closed block. a File: line with no Lines: line after it adds nothing (extract.py keeps files from spans)"""
    blocks = PATCH_RX.findall(message or "")
    if not blocks:
        return None
    out, cur = {}, ""
    for raw in blocks[-1].splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("File:"):
            cur = norm_pred_path(line[len("File:"):].strip()) or ""
            continue
        if line.startswith("Lines:") and cur:
            m = re.match(r"(\d+)\s*-\s*(\d+)", line[len("Lines:"):].strip())
            if not m:
                continue
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            out.setdefault(cur, []).append((a, b))
    return out


def declared(message: str, repo: str) -> tuple[set, dict, dict]:
    """(files, merged lines, info) after the official path filter. lines with a start or end below 1 are dropped
    (evaluate.py:_step_lines)"""
    spans = parse_patch_context(message)
    info = {"found": spans is not None, "unclosed": spans is None and "<PATCH_CONTEXT>" in (message or ""),
            "dropped_paths": []}
    files, lines = set(), {}
    for p, iv in (spans or {}).items():
        rel = resolve(repo, p)
        if not rel:
            info["dropped_paths"].append(p)
            continue
        files.add(rel)
        for a, b in iv:
            if a > 0 and b > 0:
                lines.setdefault(rel, []).append((a, b))
    return files, {f: merge(v) for f, v in lines.items()}, info


# reached context

NUMBERED = re.compile(r"^\s*(\d+)\t(.*)$")
HEADER = re.compile(r"^--- (.+?):(\d+)-(\d+)(?:\s.*)?$")
BASH_WITHHELD = re.compile(r"^\[(?:bash output|output) lines? (\d+)(?:-(\d+))? withheld\b.*\]$")
NOTE = re.compile(r"^\[(?:output cut|read cut|bash output|output lines)\b.*\]$")
COUNT = re.compile(r"^(?:read|bash): (?:\d+ of \d+ chunks|nothing in |\d+ parts? (?:is|are) withheld|.* is short \()")
EXIT = re.compile(r"^\(exit code -?\d+\)$")
GREP_SEP = re.compile(r"[:-](\d+)[:-]")
TRACEBACK = re.compile(r'^\s*File "([^"]+)", line (\d+)')
NO_CONTENT = {"ls", "pwd", "wc", "echo", "which", "tree", "du", "stat", "file", "mkdir", "touch", "cd", "true",
              "type", "whoami", "date", "env"}
PRINTERS = {"cat", "nl", "head", "tail", "sed"}


class Files:
    """the reference checkout (the task's pristine repo/) with the two line counts: \\n lines, as git, grep -n,
    sed and cat -n count them, and the read tool's lines (str.splitlines, which also breaks on \\r, \\f, \\v and a
    few unicode marks). gold and all scores use \\n lines; a read line is mapped to its \\n line"""

    def __init__(self, repo: str):
        self.repo = repo
        self._cache = {}
        self._found = {}
        self._index = {}
        self._all = None

    def get(self, rel: str):
        if rel not in self._cache:
            raw = Path(self.repo, rel).read_bytes().decode("utf-8", errors="replace")
            nl = raw.split("\n")
            if raw.endswith("\n"):
                nl = nl[:-1]
            read_lines, read_map, before = [], [], 0
            for seg in raw.splitlines(keepends=True):
                read_lines.append(seg.rstrip("\r\n\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"))
                read_map.append(before + 1)
                before += seg.count("\n")
            self._cache[rel] = (nl, read_lines, read_map)
        return self._cache[rel]

    def equal_lines(self, rel: str, shown: str) -> list[int]:
        """the \\n lines of rel whose text, stripped, equals shown stripped"""
        if rel not in self._index:
            idx = {}
            for k, t in enumerate(self.get(rel)[0], 1):
                idx.setdefault(t.strip(), []).append(k)
            self._index[rel] = idx
        key = shown.strip()
        return self._index[rel].get(key, []) if key else []

    def all(self) -> list[str]:
        if self._all is None:
            out = []
            for root, dirs, files in os.walk(self.repo):
                dirs[:] = [d for d in dirs if d != ".git"]
                out += [os.path.relpath(os.path.join(root, f), self.repo).replace(os.sep, "/") for f in files]
            self._all = sorted(out)
        return self._all

    def find(self, path: str) -> str:
        """a path as the agent or a tool wrote it: official suffix resolution, then a unique repo file ending with
        it (a grep run from a subdirectory)"""
        if path not in self._found:
            rel = resolve(self.repo, path)
            p = path.strip().lstrip("./")
            if not rel and p:
                hits = [f for f in self.all() if f.endswith("/" + p)]
                rel = hits[0] if len(hits) == 1 else ""
            self._found[path] = rel
        return self._found[path]


def _same(shown: str, line: str) -> bool:
    # whole-output strip in the bash tool eats the first row's indent; grep -o shows part of a line
    s = shown.strip()
    return s == line.strip() or (bool(s) and s in line)


def _rows(content: str):
    """(row number, text) for the output rows of one bash result, harness lines taken out. a `--- name:a-b` header
    restarts the numbering at a, a withheld row skips its range"""
    rows, n = [], 1
    for line in content.split("\n"):
        if EXIT.match(line) or COUNT.match(line):
            continue
        m = HEADER.match(line)
        if m:
            n = int(m.group(2))
            continue
        m = BASH_WITHHELD.match(line)
        if m:
            n = int(m.group(2) or m.group(1)) + 1
            continue
        if NOTE.match(line):
            continue
        rows.append((n, line))
        n += 1
    return rows


def _grep_row(text: str, files: Files):
    """(path, line, shown) for a grep-style row path:n:text or path-n-text, the path checked against the repo"""
    for m in GREP_SEP.finditer(text):
        path = text[:m.start()]
        if not path or path.endswith(" "):
            continue
        rel = files.find(path)
        if rel:
            return rel, int(m.group(1)), text[m.end():]
    return None


def _split_cmd(command: str):
    """the stages of a single pipeline, as token lists. a leading `cd dir &&` and stderr redirects are dropped.
    None for anything else (;, &&, ||, redirects to files, subshells)"""
    c = re.sub(r"\s2>\s*/dev/null|\s2>&1", " ", command.strip())
    c = re.sub(r"^cd\s+\S+\s*&&\s*", "", c)
    try:
        lex = shlex.shlex(c, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError:
        return None
    stages, cur = [], []
    for t in toks:
        if t == "|":
            stages.append(cur)
            cur = []
        elif t and set(t) <= set(";&|<>()"):
            return None
        else:
            cur.append(t)
    stages.append(cur)
    return stages if all(stages) else None


def _count_opt(args: list[str], default: int):
    """head/tail count: -n N, -nN, -N, --lines=N, and tail's +K (from line K). returns (n, plus)"""
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-n", "--lines") and i + 1 < len(args):
            v = args[i + 1]
            return (int(v.lstrip("+")), v.startswith("+")) if v.lstrip("+").isdigit() else (None, False)
        if a.startswith("--lines="):
            v = a.split("=", 1)[1]
            return (int(v.lstrip("+")), v.startswith("+")) if v.lstrip("+").isdigit() else (None, False)
        if re.fullmatch(r"-n\+?\d+", a):
            return int(a[2:].lstrip("+")), "+" in a
        if re.fullmatch(r"-\d+", a):
            return int(a[1:]), False
        i += 1
    return default, False


def _sed_range(script: str):
    m = re.fullmatch(r"(\d+)(?:,(\d+|\$))?p", script.strip())
    if not m:
        return None
    a = int(m.group(1))
    b = m.group(2)
    return a, (None if b == "$" else int(b) if b else a)


def _operands(args: list[str], takes_value: set[str]) -> list[str]:
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a in takes_value:
            i += 2
            continue
        if a.startswith("-") and a != "-":
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _printed(stages, files: Files):
    """for a pipeline that prints part of one file: (path, file lines in output order, by_number, strip_num).
    by_number: the rows carry the file line number (cat -n or nl as the first stage). strip_num: rows start with a
    number to strip (a later cat -n or nl numbers the stream, not the file). None if not such a command"""
    prog, args = stages[0][0].rsplit("/", 1)[-1], stages[0][1:]
    if prog not in PRINTERS:
        return None
    numbered = False
    if prog == "sed":
        if "-n" not in args and not any(re.fullmatch(r"-n\w*", a) for a in args):
            return None
        ops = _operands([a for a in args if a != "-e"], set())
        if len(ops) != 2:
            return None
        rng, path = _sed_range(ops[0]), ops[1]
        if rng is None:
            return None
    else:
        ops = _operands(args, {"-n", "--lines", "-v", "-w", "-s", "-b", "-i"} if prog in ("nl",) else
                        {"-n", "--lines"} if prog in ("head", "tail") else set())
        if len(ops) != 1:
            return None
        path = ops[0]
        numbered = prog == "nl" or (prog == "cat" and any(a in ("-n", "-b") for a in args))
    rel = files.find(path)
    if not rel:
        return None
    n_lines = len(files.get(rel)[0])
    if prog == "sed":
        a, b = rng
        seq = list(range(a, min(b if b is not None else n_lines, n_lines) + 1))
    elif prog == "head":
        k, _ = _count_opt(args, 10)
        if k is None:
            return None
        seq = list(range(1, min(k, n_lines) + 1))
    elif prog == "tail":
        k, plus = _count_opt(args, 10)
        if k is None:
            return None
        seq = list(range(k, n_lines + 1)) if plus else list(range(max(1, n_lines - k + 1), n_lines + 1))
    else:
        seq = list(range(1, n_lines + 1))
    renumbered = False
    for st in stages[1:]:
        p, a = st[0].rsplit("/", 1)[-1], st[1:]
        if p == "head":
            k, _ = _count_opt(a, 10)
            if k is None:
                return None
            seq = seq[:k]
        elif p == "sed" and "-n" in a:
            ops = _operands([x for x in a if x != "-e"], set())
            rng = _sed_range(ops[0]) if len(ops) == 1 else None
            if rng is None:
                return None
            seq = seq[rng[0] - 1:(rng[1] if rng[1] is not None else len(seq))]
        elif p in ("cat", "nl") and not _operands(a, set()):
            renumbered = True
        else:
            return None
    return rel, seq, numbered and not renumbered, renumbered or numbered


GREP_PROGS = {"grep", "egrep", "fgrep", "rg"}
GREP_LONG_VALUE = {"--regexp", "--file", "--max-count", "--include", "--exclude", "--exclude-dir", "--context",
                   "--after-context", "--before-context", "--glob", "--type"}
GREP_NO_TEXT = {"l", "L", "c", "q", "--count", "--files-with-matches", "--files-without-match", "--quiet"}


def _grep_args(prog: str, args: list[str]) -> tuple[set, list[str], bool]:
    """(flags, operands, pattern given by -e or -f) for a grep-family command. short flags may be combined (-nC 2,
    -nC2); a value-taking one ends the cluster"""
    takes = "ABCmef" + ("gt" if prog == "rg" else "")
    flags, ops, has_e, i = set(), [], False, 0
    while i < len(args):
        a = args[i]
        if a == "--":
            ops += args[i + 1:]
            break
        if a.startswith("--"):
            name = a.split("=", 1)[0]
            flags.add(name)
            has_e = has_e or name in ("--regexp", "--file")
            if "=" not in a and name in GREP_LONG_VALUE:
                i += 1
        elif a.startswith("-") and len(a) > 1:
            for k, ch in enumerate(a[1:]):
                flags.add(ch)
                has_e = has_e or ch in "ef"
                if ch in takes:
                    if k == len(a) - 2:
                        i += 1  # the value is the next token
                    break
        else:
            ops.append(a)
        i += 1
    return flags, ops, has_e


def _grep_single(stages, files: Files):
    """the one file a non-recursive grep reads, when rows come without a path"""
    prog = stages[0][0].rsplit("/", 1)[-1]
    if prog not in GREP_PROGS:
        return None
    flags, ops, has_e = _grep_args(prog, stages[0][1:])
    if flags & {"r", "R", "--recursive"}:
        return None
    fops = ops if has_e else ops[1:]
    if len(fops) != 1:
        return None
    return files.find(fops[0]) or None


def bash_reached(command: str, content: str, files: Files) -> tuple[dict, str, dict]:
    """(file -> set of \\n lines, status, counts). status: exact (every content row placed), partial (some),
    unknown (content, none placed), none (no file content)"""
    rows = [(n, t) for n, t in _rows(content) if t.strip() and t.strip() != "--"]
    got: dict[str, set] = {}
    counts = {"rows": len(rows), "placed": 0, "mismatch": 0, "by_content": 0}
    # a harness error (no question, bad arguments, timeout) has no exit code line
    if not rows or (content.startswith("Error:") and not any(EXIT.match(x) for x in content.split("\n"))):
        return got, "none", counts
    stages = _split_cmd(command)
    first = stages[0][0].rsplit("/", 1)[-1] if stages else ""
    unplaced = []
    # grep-style rows name their file
    for n, t in rows:
        g = _grep_row(t, files)
        if g is None:
            unplaced.append((n, t))
            continue
        rel, line, shown = g
        nl = files.get(rel)[0]
        if 1 <= line <= len(nl) and _same(shown, nl[line - 1]):
            got.setdefault(rel, set()).add(line)
            counts["placed"] += 1
        else:
            counts["mismatch"] += 1
    # python tracebacks: a File "path", line n row, then the source line
    if unplaced:
        rest, skip = [], set()
        for k, (n, t) in enumerate(unplaced):
            if k in skip:
                continue
            m = TRACEBACK.match(t)
            rel = files.find(m.group(1)) if m else ""
            if rel and k + 1 < len(unplaced):
                line, nl = int(m.group(2)), files.get(rel)[0]
                if 1 <= line <= len(nl) and _same(unplaced[k + 1][1], nl[line - 1]):
                    got.setdefault(rel, set()).add(line)
                    counts["placed"] += 2
                    skip.add(k + 1)
                    continue
            rest.append((n, t))
        unplaced = rest
    if unplaced and stages:
        single = _grep_single(stages, files)
        if single:
            nl = files.get(single)[0]
            rest = []
            for n, t in unplaced:
                m = re.match(r"^\s*(\d+)[:-](.*)$", t)
                if m and 1 <= int(m.group(1)) <= len(nl) and _same(m.group(2), nl[int(m.group(1)) - 1]):
                    got.setdefault(single, set()).add(int(m.group(1)))
                    counts["placed"] += 1
                else:
                    rest.append((n, t))
            unplaced = rest
    # grep rows with no line number (no -n, or rg): path:text, path-text, or bare text from one file. placed only
    # when the text equals exactly one line of that file
    if unplaced and stages and first in GREP_PROGS:
        single = _grep_single(stages, files)
        rest = []
        for n, t in unplaced:
            cands = [(single, t)] if single else []
            for m in re.finditer(r"[:-]", t):
                path = t[:m.start()]
                if " " in path:
                    break
                rel = files.find(path) if path else ""
                if rel:
                    cands.append((rel, t[m.end():]))
                    break
            for rel, shown in cands:
                hits = files.equal_lines(rel, shown)
                if len(hits) == 1:
                    got.setdefault(rel, set()).add(hits[0])
                    counts["placed"] += 1
                    counts["by_content"] += 1
                    break
            else:
                rest.append((n, t))
        unplaced = rest
    if unplaced and stages:
        pr = _printed(stages, files)
        if pr:
            rel, seq, by_number, strip_num = pr
            nl = files.get(rel)[0]
            hits, rest = set(), []
            for n, t in unplaced:
                shown, line = t, None
                m = NUMBERED.match(t)
                if strip_num and m:
                    shown = m.group(2)
                    if by_number:
                        line = int(m.group(1))
                if line is None and 1 <= n <= len(seq):
                    line = seq[n - 1]
                if line is not None and 1 <= line <= len(nl) and _same(shown, nl[line - 1]):
                    hits.add(line)
                else:
                    rest.append((n, t))
            # a print command whose rows mostly do not match the file is not what it looks like
            if len(rest) * 2 <= len(unplaced):
                got.setdefault(rel, set()).update(hits)
                counts["placed"] += len(unplaced) - len(rest)
                unplaced = rest
                # its blank rows count as a read's do: placed by position where the file line is blank too
                for n, t in _rows(content):
                    line = seq[n - 1] if not t.strip() and 1 <= n <= len(seq) else None
                    if line is not None and 1 <= line <= len(nl) and not nl[line - 1].strip():
                        got[rel].add(line)
                        counts["rows"] += 1
                        counts["placed"] += 1
    if not got:
        if stages and all(s[0].rsplit("/", 1)[-1] in NO_CONTENT or
                          (s[0] == "git" and len(s) > 1 and s[1] in ("status", "branch", "rev-parse")) or
                          (s[0] == "find" and "-exec" not in s and "-execdir" not in s)
                          for s in stages):
            return got, "none", counts
        if first in GREP_PROGS and _grep_args(first, stages[0][1:])[0] & GREP_NO_TEXT:
            return got, "none", counts
        return got, "unknown", counts
    return got, ("partial" if unplaced or counts["mismatch"] else "exact"), counts


def read_reached(path: str, content: str, files: Files) -> tuple[dict, str, dict]:
    """numbered rows of a read are file lines of the read's path, under any header; withheld and cut lines carry no
    numbered rows, so they add nothing"""
    counts = {"rows": 0, "placed": 0, "mismatch": 0, "by_content": 0}
    rel = files.find(path) if path else ""
    if not rel or content.startswith("Error:"):
        return {}, "none", counts
    _, read_lines, read_map = files.get(rel)
    got = set()
    for line in content.split("\n"):
        m = NUMBERED.match(line)
        if not m:
            continue
        counts["rows"] += 1
        k = int(m.group(1))
        if 1 <= k <= len(read_lines) and (m.group(2) == read_lines[k - 1] or _same(m.group(2), read_lines[k - 1])):
            got.add(read_map[k - 1])
            counts["placed"] += 1
        else:
            counts["mismatch"] += 1
    if not counts["rows"]:
        return {}, "none", counts
    if not got:
        return {}, "unknown", counts
    return {rel: got}, ("partial" if counts["mismatch"] else "exact"), counts


def reached(transcript: list[dict], repo: str) -> tuple[dict, list[dict]]:
    """what reached the agent, per tool call: file -> set of \\n lines, and one row per call with its status"""
    files = Files(repo)
    calls = {}
    for m in transcript:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls[c.get("id", "")] = (fn.get("name", ""), args if isinstance(args, dict) else {})
    got: dict[str, set] = {}
    rows = []
    for m in transcript:
        if m.get("role") != "tool":
            continue
        name, args = calls.get(m.get("tool_call_id", ""), ("", {}))
        content = m.get("content") or ""
        if isinstance(content, list):  # content parts
            content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
        if name == "read":
            g, status, counts = read_reached(str(args.get("path") or ""), content, files)
        elif name == "bash":
            g, status, counts = bash_reached(str(args.get("command") or ""), content, files)
        else:
            g, status, counts = {}, "none", {"rows": 0, "placed": 0, "mismatch": 0, "by_content": 0}
        for f, ls in g.items():
            got.setdefault(f, set()).update(ls)
        rows.append({"call_id": m.get("tool_call_id", ""), "tool": name, "status": status,
                     "lines": sum(len(v) for v in g.values()), **counts})
    return got, rows


def as_intervals(lines: set) -> list:
    out = []
    for n in sorted(lines):
        if out and n == out[-1][1] + 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


def score(task_dir: Path, work: Path, run_dir: Path | None, spec: dict) -> dict:
    repo = str(task_dir / "repo")
    gold_files, gold_lines = gold_sets(spec["gold"])
    result = json.loads((run_dir / "result.json").read_text())
    msg = result.get("final_message") or ""
    pf, pl, info = declared(msg, repo)
    d = level_scores(pf, pl, gold_files, gold_lines)
    if not info["found"] or not (pf or pl):
        # the official evaluator marks this instance an error (no_context_extracted); here it scores zero
        d = {k: (0.0 if isinstance(v, float) else v) for k, v in d.items()}
    out = {f"cb_decl_{k}": v for k, v in d.items()}
    out["cb_decl_found"] = info["found"]
    out["cb_decl_unclosed"] = info["unclosed"]
    out["cb_decl_dropped_paths"] = len(info["dropped_paths"])
    tr_path = run_dir / "transcript.json"
    rows = []
    if tr_path.exists():
        got, rows = reached(json.loads(tr_path.read_text()), repo)
        rf = {f for f, v in got.items() if v}
        rl = {f: as_intervals(v) for f, v in got.items() if v}
        r = level_scores(rf, rl, gold_files, gold_lines)
        out.update({f"cb_reach_{k}": v for k, v in r.items()})
        for s in ("exact", "partial", "unknown", "none"):
            out[f"cb_reach_calls_{s}"] = sum(1 for x in rows if x["status"] == s)
        out["cb_reach_calls"] = len(rows)
    detail = (f"declared: files {out['cb_decl_file_hit']}/{out['cb_decl_file_gold']}, lines "
              f"{out['cb_decl_line_hit']}/{out['cb_decl_line_gold']} (pred {out['cb_decl_line_pred']})")
    if not info["found"]:
        detail = "declared: no <PATCH_CONTEXT> block" + (" (not closed)" if info["unclosed"] else "")
    if "cb_reach_line_hit" in out:
        detail += (f"; reached: lines {out['cb_reach_line_hit']}/{out['cb_reach_line_gold']}, calls unknown "
                   f"{out['cb_reach_calls_unknown']}/{out['cb_reach_calls']}")
    extra = {"declared": {"files": sorted(pf), "lines": {f: v for f, v in pl.items()}, **info},
             "reached_calls": rows}
    return {"score": out["cb_decl_line_f1"], "detail": detail, **out, "_extra": extra}
