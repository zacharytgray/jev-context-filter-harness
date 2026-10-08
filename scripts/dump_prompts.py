"""writes docs/PROMPTS.md: every prompt and fixed text the experiments use, rendered from the shipped code.
the example requests come from the task data (scripts/unpack_data.py first). tests/test_prompts.py checks the doc

  uv run python scripts/dump_prompts.py           # write docs/PROMPTS.md
  uv run python scripts/dump_prompts.py --check   # exit 1 if docs/PROMPTS.md differs
"""
import argparse
import ast
import json
import re
import sys
import tempfile
import tomllib
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bench import attacks as A  # noqa: E402
from experiments.bench import common as BC  # noqa: E402
from experiments.bench import contextbench, docfinqa, legalbench  # noqa: E402
from s1harness import compaction as C  # noqa: E402
from s1harness import jev as J  # noqa: E402
from s1harness import loop as L  # noqa: E402
from s1harness import messages as M  # noqa: E402
from s1harness import pipeline as P  # noqa: E402
from s1harness import tools as T  # noqa: E402
from s1harness.levers import NAMES  # noqa: E402
from s1harness.llm import LLM  # noqa: E402

OUT = ROOT / "docs/PROMPTS.md"
TASKS = ROOT / "experiments/main/tasks.json"
# the 32K manifest of each agent model: their request settings (the 16K steps use the same)
MANIFESTS = [ROOT / f"experiments/main/manifest_{t}-32k.json" for t in ("gemma4-31b", "mistral-small-3.2-24b",
                                                                        "deepseek-v4.1-flash")]
AD = ROOT / "experiments/agentdojo"
WORDINGS_BUILD = ROOT / "experiments/attack_wordings/build.py"
ADAPTERS = {"contextbench": contextbench, "legalbenchrag": legalbench, "docfinqa": docfinqa}
BENCH_NAMES = {"contextbench": "ContextBench", "legalbenchrag": "LegalBench-RAG", "docfinqa": "DocFinQA"}
# task data blocks sit between these lines, so the test can compare the rest without the data
DATA_START, DATA_END = "<!-- from the task data -->", "<!-- end of task data -->"
REQ, QUESTION = "<the user's request>", "<the agent's question>"


def fence(text: str, lang: str = "") -> str:
    """a fenced block longer than any backtick run inside it"""
    n = max([3] + [len(m) + 1 for m in re.findall(r"`{3,}", text)])
    return f"{'`' * n}{lang}\n{text}\n{'`' * n}"


def js(obj) -> str:
    return fence(json.dumps(obj, indent=2, ensure_ascii=False), "json")


def literals(path: Path, names: set[str]) -> dict:
    """module-level constants of a file that cannot be imported here (it needs AgentDojo)"""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in names:
                out[t.id] = ast.literal_eval(node.value)
            elif isinstance(t, ast.Tuple) and any(isinstance(e, ast.Name) and e.id in names for e in t.elts):
                vals = ast.literal_eval(node.value)
                out.update({e.id: v for e, v in zip(t.elts, vals) if e.id in names})
    missing = names - set(out)
    if missing:
        raise SystemExit(f"{path.name}: no constant {', '.join(sorted(missing))}")
    return out


def must_hold(path: Path, *snippets: str):
    """the doc describes these lines of a file it cannot run. stop if they changed"""
    text = path.read_text()
    for s in snippets:
        if s not in text:
            raise SystemExit(f"{path.relative_to(ROOT)} no longer holds {s!r}: update scripts/dump_prompts.py")


# ---- jev requests, captured from the real filter functions with a fake answer


class Capture:
    def __init__(self, tmp: Path):
        self.jev = J.Jev(P.JEV_MODEL, J.Trace(tmp), api_key="unused")
        self.bodies = []
        self.jev._attempt = self._attempt

    def _attempt(self, body: dict, questions: dict):
        self.bodies.append(body)
        answers = {}
        for q, spec in questions.items():
            if spec["type"] == "noul":
                answers[q] = {"noul": 0.0}
            else:
                keys = list(spec["criteria"])
                answers[q] = {"probabilities": {k: float(k == keys[0]) for k in keys}}
        return {"model": "fake", "answers": answers}, None, answers, []


def jev_requests() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cap = Capture(Path(tmp))
        read_chunk = {"path": "<path>", "rows": [(41, "<line 41>"), (42, "<line 42>"), (43, "<line 43>")],
                      "parts": None, "numbered": True}
        bash_chunk = {"path": P.BASH_NAME, "rows": [(1, "<output line 1>"), (2, "<output line 2>")], "parts": None,
                      "numbered": False}
        P.ask_relevance(cap.jev, REQ, QUESTION, [read_chunk], command=P.read_command("<path>", 41, 43))
        P.ask_injection(cap.jev, REQ, [bash_chunk])
        P.find_span(cap.jev, REQ, {**read_chunk, "long": False})
        cap.jev.trace._jev.close()
        cap.jev.trace._llm.close()
    rel, inj, finder = cap.bodies
    return {"relevance": rel, "injection": inj, "finder": finder}


# ---- what the agent sees from the tools and the loop


def tool_replies() -> list[tuple[str, str]]:
    """real calls on a scratch dir, with placeholder names"""
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        tools = T.Tools(Path(tmp))
        (Path(tmp) / "<dir>").mkdir()
        run = lambda what, name, args: out.append((what, tools.run(name, args).text))  # noqa: E731
        run("write", "write", {"path": "<file>", "content": ""})
        run("read of an empty file", "read", {"path": "<file>"})
        run("list of an empty directory", "list", {"path": "<dir>"})
        tools.run("write", {"path": "<file>", "content": "a"})
        run("edit", "edit", {"path": "<file>", "old_text": "a", "new_text": "b"})
        run("edit whose old_text matches zero times or more than once", "edit",
            {"path": "<file>", "old_text": "<old_text>", "new_text": ""})
        run("read of a path that is not a file", "read", {"path": "<path>"})
        run("list of a path that is not a directory", "list", {"path": "<file>"})
        run("a path outside the working directory", "read", {"path": "../<path>"})
        # python's own TypeError text follows, and it varies by version
        bad = tools.run("read", {"path": "<file>", "nope": 1}).text
        out.append(("arguments the tool does not take", ": ".join(bad.split(": ", 2)[:2]) + ": <Python's message>"))
        with mock.patch.object(T.subprocess, "run", side_effect=T.subprocess.TimeoutExpired("x", P.BASH_TIMEOUT)):
            run("bash past its timeout", "bash", {"command": "<command>"})
    names = {"read", "list", "edit", "write", "bash"}
    call = lambda name, args: {"id": "c", "function": {"name": name, "arguments": args}}  # noqa: E731
    out += [("arguments that are not JSON", L._parse(call("read", "{"), names, False)[3]),
            ("arguments that are not an object", L._parse(call("read", "[]"), names, False)[3]),
            ("a tool that does not exist", L._parse(call("<name>", "{}"), names, False)[3]),
            ("read with no question (relevance on)", L._parse(call("read", '{"path": "x"}'), names, True)[3]),
            ("bash with no question (relevance on)", L._parse(call("bash", '{"command": "x"}'), names, True)[3])]
    return out


def told(role: str | None, first: str | None, n: int) -> str:
    """a read's first line once n parts are withheld, as pipeline._tell sets it"""
    items = [P._note(first, role)] if role else [P._src(1, "<line 1>")]
    return P._tell(items, n, P.short_withheld_note("<path> is", 12))[0]["text"]


def filter_notes() -> list[tuple[str, str]]:
    return [
        ("a read under the relevance filter that is short enough to pass whole",
         P.short_note("<path> is", 12)),
        ("the same for a ranged read", P.short_note("lines 10-21 of <path> are", 12)),
        ("a short read once a part of it is withheld", told("short", P.short_note("<path> is", 12), 1)),
        ("first line when the relevance filter keeps chunks (one, then several)",
         P.answer_count("read", 1, "9 chunks") + "\n" + P.answer_count("bash", 3, "9 chunks")),
        ("the withheld count, added to a read's count line (one part, then several)",
         told("count", P.answer_count("read", 2, "9 chunks"), 1) + "\n"
         + told("count", P.answer_count("read", 2, "9 chunks"), 2)),
        ("the same, as the read's own first line when it has no count line", told(None, None, 1)),
        ("the relevance filter kept nothing",
         P.nothing_kept("read", "<path>", "9 chunks", "<path>", "120-160") + "\n"
         + P.nothing_kept("bash", "the output", "9 chunks", P.BASH_NAME, "1-40")),
        ("nothing to judge (only blank lines)", P.nothing_to_judge("bash", "the output")),
        ("header above each kept chunk", P.chunk_head("<path>", "120-160")),
        ("header above a Python file's imports", P.imports_head("<path>", "1-14")),
        ("a withheld range in a read", P.read_withheld("<path>", "41-43")),
        ("the same, when the next line is shown and the planted text may go on", P.read_withheld("<path>", "41-43", True)),
        ("a withheld range in bash output (lines, then one long line)",
         P.output_withheld(P.BASH_NAME, "lines 7-12") + "\n" + P.output_withheld(P.BASH_NAME, "line 7")),
        ("the same, when the planted text may go on", P.output_withheld(P.BASH_NAME, "lines 7-12", True)),
        ("the 50,000-character cap on a read", P.cut_note("read", P.CAP_CHARS, 812, 4031)),
        ("the cap on bash output", P.cut_note("bash", P.CAP_CHARS, 812, 4031)),
    ]


# ---- the task data


def task_dir(tid: str, bench: str) -> Path:
    return BC.tasks_root() / bench / tid


def examples() -> tuple[dict, dict] | None:
    """the first clean task of each test bed in tasks.json, and the first planted one. None without the data"""
    tasks = json.loads(TASKS.read_text())["tasks"]
    clean, planted = {}, None
    for t in tasks:
        if t["kind"] == "clean" and t["benchmark"] not in clean:
            clean[t["benchmark"]] = t
        if t["kind"] == "attack" and planted is None:
            planted = t
    dirs = [task_dir(t["id"], t["benchmark"]) for t in [*clean.values(), planted]]
    if not all((d / "task.toml").is_file() for d in dirs):
        return None
    load = lambda t: tomllib.loads((task_dir(t["id"], t["benchmark"]) / "task.toml").read_text())  # noqa: E731
    return {b: (t, load(t)) for b, t in clean.items()}, (planted, load(planted))


# ---- the doc


def render(with_data: bool = True) -> str:
    ex = examples() if with_data else None
    if with_data and ex is None:
        raise SystemExit(f"no task data under {BC.tasks_root()}: run scripts/unpack_data.py first")
    o = []
    w = o.append

    w("# Prompts and fixed text")
    w("")
    w("Every prompt and every fixed piece of text in the experiments, verbatim. This file is generated from the code "
      "by `scripts/dump_prompts.py`. Do not edit it by hand. `tests/test_prompts.py` regenerates it and fails if it "
      "differs. The example requests come from the task data, so that test needs `scripts/unpack_data.py` first.")
    w("")
    w("Text in angle brackets, such as `<path>`, is a placeholder. Everything else is exactly what is sent or shown.")
    w("")

    # agent
    w("## 1. The agent's messages")
    w("")
    w("The run starts with two messages: the system prompt below, then the task's request as the user message "
      "(section 6). The system prompt is `AGENT_PROMPT`, a blank line, and `FINAL_PROMPT`, from `s1harness/messages.py`. "
      "It is the same in every configuration and every test bed.")
    w("")
    w(fence(M.system_prompt()))
    w("")
    w("Every agent call sends the messages so far and the tool list of section 2. The request settings of our runs, "
      "per agent model, as `s1harness/llm.py` builds them from each 32K manifest (the 16K step of a model uses the "
      "same):")
    w("")
    for path in MANIFESTS:
        man = json.loads(path.read_text())
        local = (man.get("endpoint") or {}).get("kind") == "local"
        base = man["endpoint"]["base_url"] if local else "https://openrouter.ai/api/v1"
        with tempfile.TemporaryDirectory() as tmp:
            trace = J.Trace(Path(tmp))
            llm = LLM(base, man["model"], trace, provider=man.get("provider"), reasoning_effort=man.get("reasoning_effort"),
                      max_output=man["max_output"], sampling=man["sampling"])
            headers = {k: v for k, v in (llm.client.default_headers or {}).items() if k == "X-Title"}
            trace._jev.close()
            trace._llm.close()
        w(f"`{path.relative_to(ROOT)}`" + (f", sent to the self-hosted server at `{base}`" if local else
                                           ", sent to OpenRouter"
                                           + (f" with the header `X-Title: {headers['X-Title']}`" if headers else "")) + ":")
        w("")
        w(js({"model": llm.model, "messages": "<the messages so far>", **llm.gen, "tools": "<section 2>", **llm.extra}))
        w("")
    w("Fields outside the OpenAI ones (`top_k`, the provider pin, the reasoning effort) go in the request body as "
      "extra fields.")
    w("")

    # tools
    w("## 2. The tools")
    w("")
    by = {}
    for (rel, inj), name in NAMES.items():
        by.setdefault(rel, []).append(f"`{name}`")
    w(f"Without the relevance filter ({', '.join(by[False])}), the agent sees these five tools, from "
      "`s1harness/tools.py:schemas`:")
    w("")
    w(js(T.schemas(relevance=False)))
    w("")
    w(f"With the relevance filter ({', '.join(by[True])}), `read` and `bash` take a required `question`, and their "
      "descriptions change:")
    w("")
    w(js(T.schemas(relevance=True)))
    w("")
    w("A run writes the list it used to `tools.json`.")
    w("")

    # jev
    reqs = jev_requests()
    w("## 3. The Jev requests")
    w("")
    w(f"Each request is one `POST {J.URL}` with a JSON body and the `TYPESAFE_API_KEY` key in the `Authorization` "
      f"header (`s1harness/jev.py`). The body is `model`, `state` and `questions`. The model name sent is "
      f"`{P.JEV_MODEL}`. A `noul` question asks for one probability of yes. A `choice` question asks for one "
      "probability per option. The bodies below are captured from the filter functions in `s1harness/pipeline.py`. "
      "Only the content is replaced by placeholders.")
    w("")
    w(f"A chunk is at most {P.CHUNK_LINES} lines and {P.CHUNK_CHARS:,} characters. A read's chunk carries its line "
      "numbers as the read shows them. Bash output goes as raw lines, with the path `bash output`. In AgentDojo the "
      "path is `<tool name> output` and the request is the user task's prompt.")
    w("")
    w(f"### Relevance filter (`ask_relevance`, kept at p >= {P.RELEVANCE_THRESHOLD})")
    w("")
    w("One request per chunk. The user's request and the agent's call ride beside the agent's question. The call "
      "(`command`) is the bash command as written, or for a read `read <path>`, with ` (lines <a>-<b>)` when the read "
      "gives a range (`read_command`).")
    w("")
    w(js(reqs["relevance"]))
    w("")
    w(f"### Injection filter (`ask_injection`, flagged at p >= {P.INJECTION_THRESHOLD})")
    w("")
    w("One request per chunk. A flagged chunk goes to the line finder, and after each span is withheld the same "
      f"question is asked again on the lines left, up to {P.INJECTION_MAX_CALLS} Jev calls per chunk "
      "(`injection_loop`).")
    w("")
    w(js(reqs["injection"]))
    w("")
    w(f"### Line finder (`line_requests`, wording `{P.FINDER_WORDING}`)")
    w("")
    w("One request with two `choice` questions over the flagged chunk's line numbers: the first and the last line of "
      f"the inserted text. A chunk over {P.CHOICE_MAX} lines is not asked; it is withheld whole.")
    w("")
    w(js(reqs["finder"]))
    w("")

    # compaction
    older = [{"role": "assistant", "content": "<the agent's text>",
              "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": '{"path": "<path>"}'}}]},
             {"role": "tool", "tool_call_id": "c1", "content": "<the tool output, as the agent saw it>"}]
    earlier = [{"role": "user", "content": C.SUMMARY_HEAD + "<the last summary>"}]
    w("## 4. Compaction")
    w("")
    w(f"Once an agent call's prompt passes the compaction threshold ({man['compaction_budget']:,} tokens, or 16,384 in "
      "the 16K run set), one call summarizes the work between the request and the newest exchange "
      "(`s1harness/compaction.py`). Its two messages:")
    w("")
    w(js([{"role": "system", "content": C.SUMMARY_SYSTEM},
          {"role": "user", "content": C.SUMMARY_USER.format(request=REQ, older="<the older work>")}]))
    w("")
    w("`<the older work>` is the last summary, if any, then the older exchanges, rendered as plain text by "
      "`compaction.render`:")
    w("")
    w(fence(C.render(earlier + older)))
    w("")
    w("After it, the second message of the context is the request and the summary in one user message "
      "(`loop.request_note`). The system prompt and the newest exchange stay as they were.")
    w("")
    w(fence(L.request_note(REQ, "<the summary>")))
    w("")

    # notes
    must_hold(ROOT / "s1harness/loop.py", 'f"(exit code {res.exit_code})"')
    w("## 5. Notes and errors the agent can see")
    w("")
    w("Harness lines the filters and the cap add to a `read` or `bash` result (`s1harness/pipeline.py`). The agent "
      "never sees a probability. Numbers here are examples.")
    w("")
    for what, text in filter_notes():
        w(f"- {what}:")
        w("")
        w(fence(text))
        w("")
    w("`bash` output ends with its exit code on its own line, after the filters (`s1harness/loop.py`):")
    w("")
    w(fence("(exit code 0)"))
    w("")
    w("Tool replies and errors (`s1harness/tools.py`, `s1harness/loop.py`):")
    w("")
    for what, text in tool_replies():
        w(f"- {what}: `{text}`")
    w("")

    # test beds
    w("## 6. Test-bed requests")
    w("")
    must_hold(ROOT / "experiments/run.py", '(base / "request.txt").write_text(task["request"])')
    must_hold(ROOT / "s1harness/cli.py", 'Path(a.request[1:]).read_text() if a.request.startswith("@")')
    w("Each task's request is its adapter's `REQUEST` template, filled in, and stored as `request` in the task's "
      "`task.toml`. `experiments/run.py` writes it to `request.txt` as it is, and the CLI reads that file back in "
      "Python's text mode. So a Windows line ending (`\\r\\n`) in an issue reaches the agent as `\\n`.")
    w("")
    if ex is not None:
        tasks = json.loads(TASKS.read_text())["tasks"]
        crlf = [t for t in tasks if "\r" in tomllib.loads(
            (task_dir(t["id"], t["benchmark"]) / "task.toml").read_text())["request"]]
        where = ", ".join(sorted({BENCH_NAMES[t["benchmark"]] for t in crlf}))
        w(DATA_START)
        w(f"{len(crlf)} of the {len(tasks)} task versions have a `\\r` in their request ({where}).")
        w(DATA_END)
        w("")
    for bench, mod in ADAPTERS.items():
        w(f"### {BENCH_NAMES[bench]} (`experiments/bench/{Path(mod.__file__).name}`)")
        w("")
        w(fence(mod.REQUEST))
        w("")
        if bench == "contextbench":
            w("Adapted from ContextBench's own ask, as the comment above `REQUEST` in the adapter says.")
            w("")
        if bench == "docfinqa":
            w(f"`{{file}}` is `{docfinqa.FILE}`.")
            w("")
        if ex is not None:
            t, toml = ex[0][bench]
            w(DATA_START)
            w(f"Example, task `{t['id']}`:")
            w("")
            w(fence(toml["request"]))
            w(DATA_END)
            w("")

    # attack
    ii = "important_instructions"
    tpl, user, model = A.ATTACKS[ii]
    w("## 7. The planted attack text")
    w("")
    w(f"Every planted task uses AgentDojo's `{ii}` template, copied verbatim from AgentDojo "
      f"(github.com/ethz-spylab/agentdojo) at commit `{A.AGENTDOJO_COMMIT}`, benchmark version "
      f"`{A.AGENTDOJO_VERSION}`: `{A.PROVENANCE[ii]}`. It is filled as AgentDojo fills it, with user `{user}` and model "
      f"`{model}` (`experiments/bench/attacks.py`). Python's repr, so the leading and trailing newlines show:")
    w("")
    w(fence(repr(tpl), "python"))
    w("")
    w("The goal is a harmless marker action:")
    w("")
    w(fence(A.goal("<marker path>", "<marker token>")))
    w("")
    w("In a code file the text goes in as line comments; in a contract or a filing, as a plain paragraph.")
    w("")
    if ex is not None:
        t, toml = ex[1]
        lab = toml["labels"]
        rendered = A.render(lab["attack"], lab["marker_path"], lab["marker_token"])
        if rendered != lab["attack_text"]:
            raise SystemExit(f"{t['id']}: attack_text differs from attacks.render")
        w(DATA_START)
        w(f"Example, task `{t['id']}`, rendered:")
        w("")
        w(fence(rendered))
        w("")
        w(f"As planted in `{lab['planted_file']}`, lines {lab['planted_lines'][0]}-{lab['planted_lines'][1]}, "
          f"as {lab['planted_as']}:")
        w("")
        w(fence(lab["planted_text"]))
        w(DATA_END)
        w("")

    # agentdojo
    ra = AD / "run_agentdojo.py"
    k = literals(ra, {"VERSION", "MODEL", "PROVIDER", "TEMPERATURE", "TOP_P", "TOP_K", "MAX_TOKENS", "PIPE_PREFIX",
                      "ATTACKS"})
    must_hold(ra, "SystemMessage(load_system_message(None))", "InitQuery()", "ToolsExecutionLoop(loop)",
              '"str": tool_result_to_str')
    ad = json.loads((AD / "manifest.json").read_text())
    yml = json.loads((AD / "manifest_travel_yaml.json").read_text())
    names = k["ATTACKS"][ad["attack"]]
    w("## 8. The AgentDojo run")
    w("")
    w("AgentDojo builds the agent's prompts itself. The runner (`experiments/agentdojo/run_agentdojo.py`) sets only "
      "these:")
    w("")
    w(f"- AgentDojo: `{ad['runner'].split('(', 1)[1].rstrip(')')}`.")
    w("- System prompt: AgentDojo's own default, `load_system_message(None)`, from `src/agentdojo/data/"
      "system_messages.yaml` in that package. It is not copied here. Each episode's AgentDojo log in "
      "`data/agentdojo-gemma4-31b.tar.zst` and `data/agentdojo-deepseek-v4.1-flash.tar.zst` holds it as sent.")
    w("- User message: the user task's prompt, as AgentDojo's `InitQuery` sends it.")
    w("- Tools: each suite's AgentDojo tools, in AgentDojo's OpenAI format.")
    w(f"- Attack: `{ad['attack']}`, rendered by AgentDojo with user `{names[0]}` and model `{names[1]}`. The model "
      f"name comes from the pipeline name, `{k['PIPE_PREFIX']}_<defense>`.")
    w(f"- Tool output: AgentDojo's `tool_result_to_str` for banking, slack and workspace; "
      f"`{' '.join(yml['parts']['a_travel_yaml']['extra_args'])}` for travel.")
    w(f"- Model `{k['MODEL']}`, provider `{json.dumps(k['PROVIDER'])}`, temperature {k['TEMPERATURE']}, top_p "
      f"{k['TOP_P']}, top_k {k['TOP_K']}, max_tokens {k['MAX_TOKENS']:,}.")
    # the deepseek run: the same runner with --llm-config. configure() pins a named provider as above
    ds = json.loads((AD / "deepseek_llm_config.json").read_text())
    dsm = json.loads((AD / "manifest_deepseek-v4.1-flash.json").read_text())
    must_hold(ra, 'PIPE_PREFIX = "openai-compatible_" + MODEL.split("/")[-1]',
              '**({"reasoning": {"effort": REASONING_EFFORT}} if REASONING_EFFORT else {})')
    if dsm["attack"] != ad["attack"]:
        raise SystemExit("the DeepSeek AgentDojo manifest names another attack: update scripts/dump_prompts.py")
    pin = {"order": [ds["provider"]], "only": [ds["provider"]], "allow_fallbacks": False}
    s = ds["sampling"]
    w(f"- The DeepSeek V4.1 Flash run (`{AD.relative_to(ROOT).as_posix()}/manifest_deepseek-v4.1-flash.json`) is the "
      f"same with `--llm-config deepseek_llm_config.json`: model `{ds['model']}`, provider `{json.dumps(pin)}`, "
      f"temperature {s['temperature']}, top_p {s['top_p']}, no top_k, max_tokens {s['max_tokens']:,}, and "
      f"`reasoning.effort` `{ds['reasoning_effort']}`. Its pipeline name is "
      f"`openai-compatible_{ds['model'].split('/')[-1]}_<defense>`, so the attack names the same user and model. "
      f"The reasoning is logged in `llm_calls.jsonl` and never sent back.")
    w("- The `jev` defense sends the injection and line-finder requests of section 3, with the user task's prompt "
      "as `request`, and shows the agent the withheld lines of section 5.")
    w("")

    # wordings
    tw = literals(WORDINGS_BUILD, {"TEMPLATES"})["TEMPLATES"]
    w("## 9. The attack-wordings check")
    w("")
    w(f"`experiments/attack_wordings/build.py` plants five AgentDojo templates (`TEMPLATES`), each from AgentDojo at "
      f"`{A.AGENTDOJO_COMMIT[:8]}` (`{A.AGENTDOJO_VERSION}`) as `experiments/bench/attacks.py` copies it. The check "
      "asks only Jev, with the injection and line-finder requests of section 3.")
    w("")
    for name in tw:
        tpl, user, model = A.ATTACKS[name]
        w(f"### `{name}`")
        w("")
        w(f"Source: `{A.PROVENANCE[name]}`. User `{user}`, model `{model}`.")
        w("")
        w(fence(repr(tpl), "python"))
        w("")
        if "{tools}" in tpl:
            w("`{tools}` is filled with one sentence naming the harness's `write` tool (`attacks.tools_sentence`):")
            w("")
            w(fence(A.tools_sentence("<marker path>", "<marker token>")))
            w("")
    return "\n".join(o).rstrip("\n") + "\n"


def strip_data(text: str) -> str:
    """the doc without its task data blocks"""
    return re.sub(rf"{re.escape(DATA_START)}\n.*?{re.escape(DATA_END)}\n\n", "", text, flags=re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with docs/PROMPTS.md instead of writing it")
    a = ap.parse_args()
    text = render()
    if a.check:
        same = OUT.exists() and OUT.read_text() == text
        print("docs/PROMPTS.md is up to date" if same else "docs/PROMPTS.md differs: run scripts/dump_prompts.py")
        sys.exit(0 if same else 1)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
