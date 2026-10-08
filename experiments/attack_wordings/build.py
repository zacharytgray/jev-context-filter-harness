"""build the template check's inputs: items.jsonl and manifest.json. no LLM, no Jev, no network.

set A: every distinct tool output that held AgentDojo's important_instructions injection in the AgentDojo run (filter
arm, last play of each episode). each one is rebuilt by replaying the episode's logged tool calls through AgentDojo's
own environment and ToolsExecutor, once per template (the same injection task, the same vectors) and once with no
injection (the vectors' default text). the important_instructions replay must equal the text the filter saw in that run,
or the output is left out.

set B: the 75 planted versions of the main run sets (experiments/main/tasks.json). the planted file is rebuilt with
each template at the same spot, planted the same way (experiments/bench/attacks.py), with the same goal: create
receipts/<item>.txt with the task's token. the important_instructions rebuild must equal the planted file byte for
byte. each version is read whole by the read tool (s1harness.tools) and cut by the harness's chunker
(pipeline.output_chunks, kind read). the chunks that hold an injected line are kept, and from the clean version the
chunks around the plant spot.

run with the AgentDojo venv (experiments/agentdojo/README.md), from the repo root:
  PYTHONPATH=$PWD $V experiments/attack_wordings/build.py
"""
import argparse
import copy
import difflib
import hashlib
import importlib.util
import json
import random
import sys
import tempfile
import time
import tomllib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentdojo.agent_pipeline import AgentPipeline, ToolsExecutor  # noqa: E402
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from agentdojo.types import ChatAssistantMessage, get_text_content_as_str  # noqa: E402

from experiments.bench import attacks as A  # noqa: E402
from experiments.bench import common as C  # noqa: E402
from s1harness import pipeline as P  # noqa: E402
from s1harness.tools import Tools  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_agentdojo", ROOT / "experiments/agentdojo/run_agentdojo.py")
RA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RA)

# the reference first, then the four others. tool_knowledge is close to important_instructions, a control
TEMPLATES = ("important_instructions", "tool_knowledge", "ignore_previous", "system_message", "injecagent")
REF = TEMPLATES[0]
AD_RUN = ROOT / "experiments/runs-agentdojo-gemma4-31b"
# the run's folders. travel is travel-yaml, the reported one: dicts written as yaml
AD_SUITES = {"banking": "banking", "slack": "slack", "travel": "travel-yaml", "workspace": "workspace"}
BAD_ARGS = "Invalid arguments for"  # the runner's error for a call it did not run
TASKS_JSON = ROOT / "experiments/main/tasks.json"
MARKS = ("#", "//", "--", "..")
SEED = 20261001  # orders the cases; a sample, if one is needed, is a prefix of that order
GROUPS = ("banking", "slack", "travel", "workspace", "contextbench", "legalbenchrag", "docfinqa")


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def text_line(t: str) -> bool:
    """a line that carries text: not blank, not a bare comment mark"""
    s = t.strip()
    for m in MARKS:
        if s.startswith(m):
            s = s[len(m):].strip()
            break
    return bool(s)


def changed(base: list[str], new: list[str]) -> list[int]:
    """1-based numbers of the lines of new that are not in base (render_offline.changed_lines)"""
    out = []
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(None, base, new, autojunk=False).get_opcodes():
        if tag in ("replace", "insert"):
            out.extend(range(j1 + 1, j2 + 1))
    return out


# ---- set A: the AgentDojo run

def agentdojo_targets(suite: str) -> list[dict]:
    """every filter-arm tool output of the last play that held the injection, in file order"""
    d = AD_RUN / AD_SUITES[suite]
    last = {}
    for line in (d / "records.jsonl").open():
        r = json.loads(line)
        if r["defense"] == "jev" and r["attack"] == REF:
            last[(r["user_task"], r["injection_task"])] = r
    plays = {r["play"]: k for k, r in last.items()}
    out = []
    for line in (d / "defense.jsonl").open():
        r = json.loads(line)
        if r["play"] not in plays or r.get("defense") != "jev-injection":
            continue
        if "INFORMATION>" not in r["before"] or r["shown"] != "content":
            continue
        ut, it = plays[r["play"]]
        out.append({"suite": suite, "user_task": ut, "injection_task": it, "play": r["play"], "call": r["call"],
                    "call_id": r["call_id"], "tool": r["tool"], "before": r["before"], "agentdojo_flagged": r["flagged"],
                    "agentdojo_lines_withheld": r["lines_withheld"], "agentdojo_chunks": r["chunks"],
                    "log": d / last[(ut, it)]["log"]})
    return out


def replay(suite, ut, injections: dict, messages: list[dict], fmt, want: set) -> dict:
    """the episode's tool calls, in order, on a fresh environment with these injections, through AgentDojo's
    ToolsExecutor. returns call id -> (content, error) for every call run, up to the last one in want"""
    env = ut.init_environment(suite.load_and_inject_default_environment(injections))
    runtime = FunctionsRuntime(suite.tools)
    ex = ToolsExecutor(tool_output_formatter=fmt)
    errors = {m["tool_call_id"]: m.get("error") or "" for m in messages if m["role"] == "tool"}
    out = {}
    for m in messages:
        if m["role"] != "assistant" or not m.get("tool_calls"):
            continue
        for tc in m["tool_calls"]:
            if errors.get(tc["id"], "").startswith(BAD_ARGS):  # the runner never ran it
                continue
            call = FunctionCall(**copy.deepcopy(tc))
            _, _, env, got, _ = ex.query("", runtime, env,
                                         [ChatAssistantMessage(role="assistant", content=None, tool_calls=[call])])
            t = got[-1]
            out[tc["id"]] = (get_text_content_as_str(t["content"] or []), t["error"])
        if want <= set(out):
            break
    return out


_W = {}


def _suite(name: str):
    """the suite and its five attacks, loaded once per process"""
    if name not in _W:
        suite = get_suite(RA.VERSION, name)
        ref = AgentPipeline([])
        ref.name = RA.PIPE_PREFIX  # {model} is "AI assistant", as in the run
        _W[name] = (suite, {n: load_attack(n, suite, ref) for n in TEMPLATES})
    return _W[name]


def replay_episode(job: tuple) -> dict | None:
    """one episode replayed once per template and once clean. None if the reference injections differ from the
    ones the run logged"""
    suite_name, ut_id, it_id, log_path, want, fmt_name = job
    suite, atks = _suite(suite_name)
    ut, it = suite.get_user_task_by_id(ut_id), suite.get_injection_task_by_id(it_id)
    logd = json.loads(Path(log_path).read_text())
    inj = {n: atks[n].attack(ut, it) for n in TEMPLATES}
    if inj[REF] != logd["injections"]:
        return None
    fmt = RA.TOOL_FORMATS[fmt_name]
    got = {n: replay(suite, ut, inj[n], logd["messages"], fmt, set(want)) for n in TEMPLATES}
    got["clean"] = replay(suite, ut, {}, logd["messages"], fmt, set(want))
    return got


def build_a(workers: int, log=print) -> tuple[list[dict], dict]:
    items, stats = [], {"suites": {}}
    clean_ids = {}
    for suite_name, folder in AD_SUITES.items():
        t0 = time.monotonic()
        suite, atks = _suite(suite_name)
        # the tool format the run played with, from its records (the runner's call log is not shipped)
        fmts = {json.loads(x).get("tool_format") or "str"
                for x in (AD_RUN / folder / "records.jsonl").read_text().splitlines() if x.strip()}
        if len(fmts) != 1:
            raise SystemExit(f"{folder}: more than one tool format in its records: {sorted(fmts)}")
        fmt_name = fmts.pop()
        for name in TEMPLATES:
            RA.check_names(name, atks[name])
        targets = agentdojo_targets(suite_name)
        seen, distinct = set(), []
        for t in targets:
            k = (t["user_task"], t["injection_task"], t["before"])
            if k not in seen:
                seen.add(k)
                distinct.append(t)
        by_ep = defaultdict(list)
        for t in distinct:
            by_ep[(t["user_task"], t["injection_task"])].append(t)
        st = Counter(outputs=len(targets), distinct=len(distinct))
        eps = list(by_ep.items())
        jobs = [(suite_name, ut_id, it_id, str(ts[0]["log"]), sorted(t["call_id"] for t in ts), fmt_name)
                for (ut_id, it_id), ts in eps]
        with ProcessPoolExecutor(workers) as ex:
            replays = list(ex.map(replay_episode, jobs, chunksize=4))
        for ((ut_id, it_id), ts), got in zip(eps, replays):
            ut, it = suite.get_user_task_by_id(ut_id), suite.get_injection_task_by_id(it_id)
            if got is None:
                st["injections differ from the log"] += len(ts)
                continue
            for t in ts:
                cid = t["call_id"]
                if any(cid not in g for g in got.values()):
                    st["call not replayed"] += 1
                    continue
                if got[REF][cid] != (t["before"], None):
                    st["replay differs from the AgentDojo run"] += 1
                    continue
                if any(g[cid][1] for g in got.values()):
                    st["a version returned an error"] += 1
                    continue
                clean = got["clean"][cid][0]
                if "INFORMATION>" in clean:
                    st["clean replay holds the injection"] += 1
                    continue
                # one agent turn can make several tool calls: the call id's tail keeps the case id unique
                case = f"A-{suite_name}-{ut_id}-{it_id}-c{t['call']}-{cid[-6:]}"
                name = f"{t['tool']} output"
                ckey = (suite_name, ut_id, name, clean)
                if ckey not in clean_ids:
                    clean_ids[ckey] = f"A-{suite_name}-{ut_id}-clean-{len([k for k in clean_ids if k[:2] == ckey[:2]])}"
                    items.append({"id": clean_ids[ckey], "set": "A", "case": None, "version": "clean",
                                  "group": suite_name, "kind": "text", "request": ut.PROMPT, "name": name,
                                  "text": clean, "injected_lines": [], "injected_all": [],
                                  "source": {"suite": suite_name, "user_task": ut_id, "tool": t["tool"],
                                             "first_case": case}})
                    st["clean outputs"] += 1
                base = clean.split("\n")
                versions = {}
                for name_t in TEMPLATES:
                    text = got[name_t][cid][0]
                    rows = text.split("\n")
                    ch = changed(base, rows)
                    versions[name_t] = (text, [n for n in ch if text_line(rows[n - 1])], ch)
                if any(not v[1] for v in versions.values()):
                    st["a version with no injected line"] += 1
                    continue
                for name_t, (text, inj_lines, inj_all) in versions.items():
                    items.append({"id": f"{case}-{name_t}", "set": "A", "case": case, "version": name_t,
                                  "group": suite_name, "kind": "text", "request": ut.PROMPT, "name": name,
                                  "text": text, "injected_lines": inj_lines, "injected_all": inj_all,
                                  "clean_id": clean_ids[ckey],
                                  "source": {"suite": suite_name, "user_task": ut_id, "injection_task": it_id,
                                             "goal": it.GOAL, "tool": t["tool"], "agentdojo_play": t["play"],
                                             "agentdojo_call": t["call"], "agentdojo_call_id": cid,
                                             "agentdojo_flagged": t["agentdojo_flagged"],
                                             "agentdojo_lines_withheld": t["agentdojo_lines_withheld"],
                                             "agentdojo_chunks": t["agentdojo_chunks"],
                                             "tool_format": fmt_name}})
                st["cases"] += 1
        stats["suites"][suite_name] = {**st, "tool_format": fmt_name, "seconds": round(time.monotonic() - t0, 1)}
        log(f"A {suite_name}: {dict(st)} ({stats['suites'][suite_name]['seconds']} s)", flush=True)
    return items, stats


# ---- set B: the full run's planted test beds

def plant_variant(lab: dict, planted: str, clean: str, template: str) -> tuple[str, str, list[str]]:
    """(the planted file rebuilt with this template, the text planted, the lines planted). code files get comment lines with the
    plant's indent, text documents a plain paragraph, at the same spot, as experiments/bench builds them"""
    text = A.render(template, lab["marker_path"], lab["marker_token"])
    how = lab["planted_as"]
    if how.startswith("comment lines"):
        mark = how[how.index("(") + 1:how.index(")")]
        a, b = lab["planted_lines"]
        rows = clean.split("\n")
        at = rows[a - 1] if a - 1 < len(rows) else ""
        indent = at[:len(at) - len(at.lstrip(" \t"))]
        block = A.comment_block(text, mark, indent)
        new, _, _ = A.plant_lines(clean, a, block)
        return new, text, block
    if how == "plain paragraph":
        s, e = lab["planted_chars"]
        block = A.paragraph_block(text)
        return planted[:s] + block + planted[e:], text, block.split("\n")
    raise ValueError(f"unknown plant kind {how}")


def read_chunks(tmp: Path, rel: str, content: str) -> tuple[list[tuple[int, str]], list[dict]]:
    """the whole-file read, as the read tool returns it, and the harness's chunks of it (injection on, relevance
    off: _pipeline's chunk(rows, size))"""
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content.encode("utf-8"))
    res = Tools(tmp)._read(rel)
    chunks = P.output_chunks("read", res.lines, len(res.text), rel)
    return res.lines, chunks


def keep(chunks: list[dict], lines: set[int]) -> list[dict]:
    out = []
    for i, c in enumerate(chunks):
        if lines & {n for n, _ in c["rows"]}:
            out.append({"index": i, "of": len(chunks), "path": c["path"], "rows": c["rows"], "parts": c["parts"],
                        "numbered": c["numbered"]})
    return out


def build_b(check_hash: bool, log=print) -> tuple[list[dict], dict]:
    data = C.tasks_root()
    tasks = json.loads(TASKS_JSON.read_text())["tasks"]
    sha = {t["id"]: t["sha256"] for t in tasks}
    items, st = [], Counter()
    with tempfile.TemporaryDirectory() as td:
        for t in tasks:
            if t["kind"] != "attack":
                continue
            bench = t["benchmark"]
            tdir = data / bench / t["id"]
            cid = t["id"].replace("-attack-needed", "-clean")
            cdir = data / bench / cid
            if check_hash:
                for x, d in ((t["id"], tdir), (cid, cdir)):
                    if C.task_hash(d) != sha[x]:
                        raise SystemExit(f"{x}: task dir sha256 differs from tasks.json")
            toml = tomllib.loads((tdir / "task.toml").read_text())
            lab, request = toml["labels"], toml["request"]
            rel = lab["planted_file"]
            pf, cf = tdir / "repo" / rel, cdir / "repo" / rel
            planted = pf.read_bytes().decode("utf-8")
            clean = cf.read_bytes().decode("utf-8")
            if A.render(REF, lab["marker_path"], lab["marker_token"]) != lab["attack_text"]:
                raise SystemExit(f"{t['id']}: the reference does not render as planted")
            case = "B-" + t["id"].removesuffix("-attack-needed")
            tmp = Path(td) / t["id"]
            c_rows, c_chunks = read_chunks(tmp / "clean", rel, clean)
            base = [x for _, x in c_rows]
            spot = None
            for name in TEMPLATES:
                content, text, block = plant_variant(lab, planted, clean, name)
                if name == REF and content != planted:
                    raise SystemExit(f"{t['id']}: the reference rebuild differs from the planted file")
                rows, chunks = read_chunks(tmp / name, rel, content)
                lines = [x for _, x in rows]
                ch = changed(base, lines)
                # a plant inside a line splits that line in two. both halves differ from the clean line, but they
                # are the document's own text, not injected
                planted_lines = {x.strip() for x in block}
                split = [n for n in ch if lines[n - 1].strip() not in planted_lines]
                inj = [n for n in ch if n not in split and text_line(lines[n - 1])]
                if len(inj) != sum(text_line(x) for x in block):
                    raise SystemExit(f"{t['id']} {name}: {len(inj)} changed text lines, the plant has "
                                     f"{sum(text_line(x) for x in block)}")
                spot = spot or min(n for n in ch if n not in split)  # the reference's first planted line
                tok = [n for n in inj if lab["marker_token"] in lines[n - 1]]
                kept = keep(chunks, set(inj))
                items.append({"id": f"{case}-{name}", "set": "B", "case": case, "version": name, "group": bench,
                              "kind": "read", "request": request, "path": rel, "chunks": kept,
                              "injected_lines": inj, "injected_all": [n for n in ch if n not in split],
                              "split_lines": split, "token_lines": tok,
                              "clean_id": f"{case}-clean", "planted_text": text,
                              "source": {"task": t["id"], "clean_task": cid, "task_sha256": sha[t["id"]],
                                         "clean_task_sha256": sha[cid], "file": rel, "planted_as": lab["planted_as"],
                                         "file_lines": len(rows), "file_chunks": len(chunks),
                                         "planted_file_sha256": sha_file(pf), "variant_sha256": sha_text(content)}})
                st[f"{bench} {name} chunks kept"] += len(kept)
                st[f"plants across chunks {name}"] += len(kept) > 1
                st[f"plants inside a line {name}"] += bool(split)
            # the clean version: the chunks that hold the lines on either side of the plant spot
            near = {n for n in (spot - 1, spot) if 1 <= n <= len(c_rows)}
            items.append({"id": f"{case}-clean", "set": "B", "case": None, "version": "clean", "group": bench,
                          "kind": "read", "request": request, "path": rel, "chunks": keep(c_chunks, near),
                          "injected_lines": [], "injected_all": [], "token_lines": [],
                          "source": {"task": cid, "task_sha256": sha[cid], "file": rel,
                                     "clean_file_sha256": sha_file(cf), "spot_lines": sorted(near)}})
            st["cases"] += 1
    log(f"B: {dict(st)}", flush=True)
    return items, dict(st)


def order(items: list[dict]) -> list[dict]:
    """one case at a time: its five templates, then its clean counterpart (once, the first time a case uses it).
    cases are shuffled with SEED inside each source and spread evenly through the order (case k of n at (k + 0.5) / n),
    so any prefix holds about the same share of every source"""
    cases, clean = defaultdict(list), {}
    for it in items:
        if it["case"]:
            cases[it["case"]].append(it)
        else:
            clean[it["id"]] = it
    by_group = defaultdict(list)
    for c, its in cases.items():
        by_group[its[0]["group"]].append(c)
    rng = random.Random(SEED)
    pos = []
    for g in GROUPS:
        cs = sorted(by_group[g])
        rng.shuffle(cs)
        pos += [((k + 0.5) / len(cs), GROUPS.index(g), c) for k, c in enumerate(cs)]
    out, seen = [], set()
    for _, _, c in sorted(pos):
        its = sorted(cases[c], key=lambda it: TEMPLATES.index(it["version"]))
        out += its
        if its[0]["clean_id"] not in seen:
            seen.add(its[0]["clean_id"])
            out.append(clean[its[0]["clean_id"]])
    return out


def fit(items: list[dict], limit: int, run) -> tuple[list[dict], dict | None]:
    """all items if the plan estimate is within limit. else every test bed case (set B), and AgentDojo cases drawn
    one suite at a time in turn (banking, slack, travel, workspace), each suite's cases in a SEED-shuffled order,
    until the next case would pass limit. so the four suites get the same number of cases, except one that runs
    out. a case is its five templates and its clean counterpart, so templates stay paired"""
    est = run.estimate(items)
    if est["plan"]["all"]["input_tokens"] <= limit:
        return items, None
    per = {}
    for it in items:
        j = run.DryJev(it)
        run.filter_item(j, it)
        per[it["id"]] = j.input_tokens
    cases, clean = defaultdict(list), {}
    for it in items:
        (cases[it["case"]].append(it) if it["case"] else clean.__setitem__(it["id"], it))
    paid, total = set(), 0

    def take(c) -> int:
        its = cases[c]
        extra = [] if its[0]["clean_id"] in paid else [its[0]["clean_id"]]
        return sum(per[it["id"]] for it in its) + sum(per[i] for i in extra)

    keep = []
    for c, its in cases.items():
        if its[0]["set"] == "B":
            total += take(c)
            paid.add(its[0]["clean_id"])
            keep.append(c)
    b_total = total
    rng = random.Random(SEED)
    queues = {}
    for g in GROUPS[:4]:
        cs = sorted(c for c, its in cases.items() if its[0]["group"] == g)
        rng.shuffle(cs)
        queues[g] = cs
    drawn = Counter()
    full = False
    while not full and any(queues.values()):
        for g in GROUPS[:4]:
            if not queues[g]:
                continue
            c = queues[g][0]
            cost = take(c)
            if total + cost > limit:
                full = True
                break
            queues[g].pop(0)
            total += cost
            paid.add(cases[c][0]["clean_id"])
            keep.append(c)
            drawn[g] += 1
    keep = set(keep)
    out = [it for it in items if (it["case"] in keep) or (not it["case"] and it["id"] in paid)]
    return order(out), {
        "rule": f"every test bed case (set B, 75), and AgentDojo cases drawn round robin over the four suites, each "
                f"suite in a random order (seed {SEED}), until the next case would take the plan estimate past "
                f"{limit:,} tokens. every template of a drawn case and its clean counterpart go in",
        "agentdojo_cases_drawn": dict(drawn),
        "agentdojo_cases_available": {g: sum(1 for its in cases.values() if its[0]["group"] == g)
                                      for g in GROUPS[:4]},
        "plan_tokens_drawn": total, "plan_tokens_set_b": b_total,
        "full_design_plan_tokens": est["plan"]["all"]["input_tokens"],
        "full_design_high_tokens": est["high"]["all"]["input_tokens"],
        "full_design_estimate": est}


def code_shas() -> dict:
    files = ["s1harness/pipeline.py", "s1harness/jev.py", "s1harness/tools.py", "experiments/bench/attacks.py",
             "experiments/agentdojo/run_agentdojo.py", "experiments/attack_wordings/build.py",
             "experiments/attack_wordings/run.py"]
    return {f: sha_file(ROOT / f) for f in files}


def agentdojo_shas() -> dict:
    """sha256 of the run files read: records and defense per suite, and one over every agentdojo log read"""
    out = {}
    for s, folder in AD_SUITES.items():
        for f in ("records.jsonl", "defense.jsonl"):
            out[f"{folder}/{f}"] = sha_file(AD_RUN / folder / f)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-task-hash", action="store_true", help="skip checking every task dir against tasks.json")
    ap.add_argument("--cache", type=Path, default=None, help="reuse sets A and B from a_set.json and b_set.json in "
                    "this folder (written if missing), to change only the order, sample or manifest")
    ap.add_argument("--workers", type=int, default=10, help="processes replaying AgentDojo episodes")
    a = ap.parse_args(argv)
    random.seed(0)  # nothing is drawn: every distinct output and every planted version goes in
    def cached(name, fn):
        p = a.cache / name if a.cache else None
        if p and p.is_file():
            return json.loads(p.read_text())
        got = fn()
        if p:
            p.write_text(json.dumps(got))
        return got

    items_a, stats_a = cached("a_set.json", lambda: build_a(a.workers))
    items_b, stats_b = cached("b_set.json", lambda: build_b(not a.no_task_hash))
    import run  # noqa: E402  the estimate is run.py's dry run

    items = order(items_a + items_b)
    items, sample = fit(items, run.PLAN_LIMIT, run)
    path = HERE / "items.jsonl"
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it, separators=(",", ":")) + "\n")
    est = run.estimate(items)
    cap = run.CAP
    manifest = {
        "name": "attack_wordings",
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": "does the injection filter catch injections worded unlike important_instructions?",
        "templates": list(TEMPLATES),
        "reference": REF,
        "agentdojo": {"commit": A.AGENTDOJO_COMMIT, "benchmark_version": RA.VERSION,
                      "names": {"user": A.USER, "model": A.MODEL}},
        "jev_model": P.JEV_MODEL,
        "filter": {"wording": P.INJECTION_WORDING, "threshold": P.INJECTION_THRESHOLD, "finder": P.FINDER_WORDING,
                   "max_calls_per_chunk": P.INJECTION_MAX_CALLS, "chunk_lines": P.CHUNK_LINES,
                   "chunk_chars": P.CHUNK_CHARS, "workers": P.INJECTION_WORKERS},
        "sampling": sample or "none: every distinct injected output of the AgentDojo run (set A) and every planted version (set B)",
        "seed": SEED,
        "order": "one case at a time (its five templates, then its clean counterpart), cases spread evenly by source",
        "inputs": {"agentdojo_dir": str(AD_RUN.relative_to(ROOT)), "agentdojo_files": agentdojo_shas(), "tasks_json": str(TASKS_JSON.relative_to(ROOT)),
                   "tasks_json_sha256": sha_file(TASKS_JSON), "task_dirs_checked": not a.no_task_hash,
                   "bench_data": str(C.data_root())},
        "code": code_shas(),
        "items": {"file": "items.jsonl", "sha256": sha_file(path), "count": len(items),
                  "by_set_version": dict(Counter(f"{i['set']} {i['version']}" for i in items))},
        "set_a": stats_a,
        "set_b": stats_b,
        "estimate": est,
        "cap": cap,
        "cap_rule": "hard stop at 19M Jev input tokens counted from Jev's usage, checked before "
                    "each request; run all if the plan estimate is at most 15M, else a stratified sample that fits",
        "price_per_1m_input": run.PRICE,
        "rate_limits": {"requests_per_s": run.RPS, "tokens_per_s": run.TPS},
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"items.jsonl: {len(items)} items, sha256 {manifest['items']['sha256']}")
    run.print_estimate(est, cap)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
