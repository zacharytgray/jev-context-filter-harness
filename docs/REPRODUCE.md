# Reproduce

Every number in the paper and its supplement, and in the `RESULTS.md` files, can be rebuilt offline from the stored
records, with no API key and no model call. Section 2 names the script for each. Rerunning the experiments themselves needs a Jev key and an agent endpoint.

## 1. Set up and unpack

```sh
uv sync
uv run pytest -q                          # offline unit tests
uv run python scripts/unpack_data.py      # checks SHA256SUMS, unpacks runs and tasks, checks each task's sha256
```

One test is skipped unless `AGENTDOJO_SRC` names an AgentDojo clone (see `experiments/agentdojo/README.md`). It checks
the attack templates in `experiments/bench/attacks.py` against AgentDojo's own source.

`data/README.md` lists what each tarball holds and where it unpacks.

The tests in `tests/test_prompts.py` read the unpacked data, so run `uv run pytest -q` again after unpacking. They
check [PROMPTS.md](PROMPTS.md) against the task requests and check recorded runs' system prompt, tools and request
against the code. To rewrite PROMPTS.md after a change to a prompt, run `uv run python scripts/dump_prompts.py`.
[VERSIONS.md](VERSIONS.md) gives the exact model, Jev, AgentDojo and package versions behind the shipped runs.

## 2. Rebuild the numbers

### The paper's tables and figure data

```sh
uv run --with numpy python experiments/main/paper_tables.py      # writes experiments/main/out/
```

It writes CSV files, plus Table 2 as `.tex`, and no plots:

- `t1_main.csv` and `t1_main.tex`: Table 2, per agent, configuration and compaction threshold.
- `f2_effects.csv`: the data of Figure 3, each configuration's paired effects against Control.
- `score_abs.csv`: each configuration's score per test bed over clean, planted and all versions: the mean, its t
  interval, the runs, and the runs that scored 0.
- `score_versions.csv`: the same cells as paired differences from Control.
- `f3_reading.csv`, `f3_tasks.csv` and `f3_rho.csv`: the data of Figure 4, raw tool output per model and per task, and
  Spearman's rho.
- `cost_ratio.csv`: the cost ratios and break-even prices of the cost section (Section 5.7).
- `t4_agentdojo.csv`: our two rows of the AgentDojo comparison table.

The score column of Table 2 and Figure 3a use the all-versions cells. `--out <folder>` writes the files elsewhere. It
needs numpy, which is not a dependency of the project, so `--with numpy` adds it for this one command. It runs the loader, the cross-model script and the cost-ratio script below in process.

### Test-bed experiment (`experiments/main/RESULTS.md`)

Each agent has one config in `experiments/main/analysis/configs/`: `gemma4-31b.json`, `mistral-small-3.2-24b.json`
and `deepseek-v4.1-flash.json`. A config names the model, its two manifests and run tags, and the expected counts.

The analysis data, per agent. This gives most numbers in RESULTS.md: clean-version scores and paired score
differences, per-task token cuts and medians, attacks followed and reached with their intervals, score under attack, compaction, wall
time, Jev tokens, and the runs that failed on every try.

```sh
python3 experiments/main/analysis/data.py --config experiments/main/analysis/configs/gemma4-31b.json
python3 experiments/main/analysis/data.py --config experiments/main/analysis/configs/mistral-small-3.2-24b.json
python3 experiments/main/analysis/data.py --config experiments/main/analysis/configs/deepseek-v4.1-flash.json
```

Each writes `experiments/main/analysis/out-<agent>/data.json` (not in git). See `experiments/main/analysis/README.md`.
Its `score` and `paired` sections are over the clean versions only. The scores over all versions come from
`paper_tables.py`.

The cross-model numbers: the token cut in mean tokens with its bootstrap interval, raw tool output per model, and
Spearman's rho between raw output and the token difference.

```sh
python3 experiments/main/cross_model/xmodel.py      # stdlib only; writes out.json and per_task.csv beside it
```

`--out <folder>` writes them elsewhere. `experiments/main/cross_model/README.md` lists what it computes. It imports
`analysis/data.py` and reads each agent's runs through the same config, so it counts the same runs. Its bootstrap and
permutation tests use 10,000 draws with seed 20261004.

The cost ratio (Jev tokens per agent prompt token saved) and the break-even prices, from the counted records:

```sh
uv run --with numpy python experiments/main/cost_ratio.py     # writes experiments/main/cost_ratio.csv
```

`--out <file>` writes the csv elsewhere. Its bootstrap uses 10,000 resamples with seed 42.

Compaction against Control and 16K against 32K (the compaction section of the paper and the supplement's compaction
rows): the share of runs that compacted, with Newcombe's paired interval, and compactions per run, with a t interval,
paired by task version over all versions:

```sh
python3 experiments/main/compaction.py [--csv <file>]      # stdlib only; prints the tables
```

The reading and latency details beside Figure 4, at 32K: runs with a whole-file read and the share of raw characters
from whole-file reads, runs with 10 or more ranged reads and the median lines per ranged read, relevance's long
outputs by class and the share of characters kept in long reads, and the median Jev request latency over the
relevance + injection runs, from `trace.jsonl`:

```sh
python3 experiments/main/cross_model/reading_details.py    # stdlib only; prints, writes nothing
```

### AgentDojo (`experiments/agentdojo/RESULTS.md`)

```sh
uv run python experiments/agentdojo/summarize.py experiments/runs-agentdojo-gemma4-31b --markdown
uv run python experiments/agentdojo/summarize.py experiments/runs-agentdojo-deepseek-v4.1-flash --markdown
```

Each prints the coverage; the three AgentDojo rates (utility without attack, utility under attack, attack success)
per suite and defense, with Wilson intervals; episode, error and token counts (agent prompt and completion tokens,
Jev input tokens); the attacks that succeeded under the filter; the cases solved without the filter but not with it;
and the errors.

The paired differences come from `agentdojo()` in `experiments/main/analysis/data.py`, which reads a run folder. Their
intervals are Newcombe's paired score interval (method 10) with Newcombe's adjustment to phi, as in the paper. Call it
directly:

```sh
cd experiments/main/analysis
python3 -c 'import data, json; print(json.dumps(data.agentdojo(data.AD_ROOT)["rows"], indent=1))'
python3 -c 'import data, json; print(json.dumps(data.agentdojo("../../runs-agentdojo-deepseek-v4.1-flash")["rows"], indent=1))'
```

The task-level facts the paper gives for DeepSeek (every stock attack success came from one travel injection task, and
five of the seven slack cases lost under attack share one user task) come from the records of the same folder:

```sh
python3 - <<'PY'
import collections, glob, json
last = {}
for f in glob.glob('experiments/runs-agentdojo-deepseek-v4.1-flash/*/records.jsonl'):
    for line in open(f):
        r = json.loads(line)
        last[(r['suite'], r['user_task'], r['injection_task'] or 'none', r['attack'], r['defense_id'])] = r
won = collections.Counter(k[:3:2] for k, r in last.items() if k[4] == 'none' and k[3] != 'none' and r['security'])
lost = collections.Counter(k[1] for k, r in last.items() if k[0] == 'slack' and k[3] != 'none' and k[4] == 'none'
                           and r['utility'] and not last[k[:4] + ('jev-injection',)]['utility'])
print('stock attack successes by suite and injection task:', dict(won))
print('slack cases lost under attack, by user task:', dict(lost))
PY
```

### Planted attack text

The length of the `important_instructions` text in the 75 planted task versions (the paper's "about 400 characters"),
without the blank lines around it, from the unpacked task data:

```sh
python3 - <<'PY'
import glob, statistics, tomllib
n = [len(tomllib.load(open(f, 'rb'))['labels']['attack_text'].strip())
     for f in sorted(glob.glob('data/tasks/*/*-attack-needed/task.toml'))]
print(f'{len(n)} planted tasks: attack text {min(n)} to {max(n)} characters, median {statistics.median(n):g}')
PY
```

It prints `75 planted tasks: attack text 386 to 486 characters, median 388`.

### Attack wordings (`experiments/attack_wordings/RESULTS.md`)

`experiments/attack_wordings/run/report.json` holds the numbers: the items and cases counted, the definition of
"instructions removed" (chosen after the results were seen), and per set (A = AgentDojo tool outputs, B = test-bed
files, A+B) and group, per template, the cases, the instructions removed (k, n, rate and Wilson 95% interval) and
`goal_fragment_left`. For the clean counterparts it holds the pooled false flags. To recompute it from the
per-version results in `data/attack-wordings.tar.zst` (unpacked to `experiments/attack_wordings/items.jsonl` and
`experiments/attack_wordings/run/`):

```sh
uv run python experiments/attack_wordings/run.py --report      # rewrites run/report.json
```

### Comparison table

`experiments/main/tables/comparison.md` and `.tex` hold the paper's AgentDojo comparison table. Our two rows come from
the AgentDojo numbers above. Every other row is copied from the cited paper's own table.

## 3. Rebuild the tasks from upstream

The shipped task dirs can be rebuilt from the upstream datasets and checked against `experiments/main/tasks.json`:

```sh
export S1H_BENCH_DATA=<a scratch folder>
PYTHONPATH=$PWD uv run python experiments/bench/fetch.py
PYTHONPATH=$PWD uv run python experiments/bench/build.py verify full --out <a scratch folder>/verify
```

`fetch.py` downloads ContextBench, LegalBench-RAG and DocFinQA at pinned revisions and checks each file's sha256. It
needs `uv` and network access. `verify full` redraws the 25 items per test bed, rebuilds every task dir under `--out`,
and checks each hash against `tasks.json`. Six task dirs will report a different hash: a rebuild keeps the upstream
fixture files named `.env` or `types.env` that the shipped dirs leave out (`data/README.md` lists them). ContextBench
repositories are fetched from GitHub at their base commits the first time a task is built.
`experiments/bench/README.md` describes the build.

## 4. Rerun the experiments

Copy `.env.example` to `.env` and set `TYPESAFE_API_KEY`, and `OPENROUTER_API_KEY` for an OpenRouter agent. The
harness CLI and the AgentDojo runner load `.env` themselves. `experiments/driver.py` and
`experiments/attack_wordings/run.py` read the keys from the environment only, so export them first, for example with
`set -a; . ./.env; set +a`.

### Test beds

With an OpenRouter agent, here Gemma 4 31B:

```sh
uv run python experiments/run.py main --arms plain,relevance,injection,relevance-injection \
    --repeats 1 --workers 12 --tag <new tag> \
    --model google/gemma-4-31b-it --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
    --provider Novita --max-output 8192 --temperature 1.0 --top-p 0.95 --top-k 64 \
    --max-turns 40 --compaction-budget 32768 --note "why this run"
```

For DeepSeek V4.1 Flash, use `--model deepseek/deepseek-v4.1-flash --provider Morph --temperature 1.0 --top-p 0.95
--reasoning-effort low` and no `--top-k`.

With a self-hosted agent, here Mistral Small 3.2 24B on vLLM (the server's version and flags are in
[VERSIONS.md](VERSIONS.md)):

```sh
uv run python experiments/run.py main --arms plain,relevance,injection,relevance-injection \
    --repeats 1 --workers 8 --tag <new tag> \
    --model mistral-small-3.2-24b --base-url http://127.0.0.1:8000/v1 --max-output 8192 --context-window 65536 \
    --temperature 0.15 --max-turns 40 --compaction-budget 32768 --note "why this run"
```

`--base-url` and `--model` are required: the harness has no default endpoint or model. Leave out `--api-key-env`
when the server needs no key. Each command plays all 600 runs of the 32K design into `experiments/runs-<new tag>/main/`.
Use a new tag so the shipped records are not touched. `--dry-run` lists the runs and stops. `--skip-done` continues
after a stop. For the 16K design, pass `--compaction-budget 16384`, or play only the runs a shipped 16K manifest lists
(see `experiments/main/README.md`). `experiments/driver.py` launches a frozen manifest instead; it has an OpenRouter
mode and a local endpoint mode. With it, a run that ends in an error is played again, up to four tries, and the last
play stands.

Agent workdirs go to `/private/tmp/s1h-eval/<tag>/` on macOS and `/tmp/s1h-eval/<tag>/` elsewhere. Set
`S1H_WORK_ROOT` to move them. They stay outside the repository so the agent cannot reach the task checks. Each run gets
3,600 s.

### AgentDojo

See `experiments/agentdojo/README.md` for the venv and the commands. Per suite there are two calls (no attack, then
`important_instructions`), with `--defense none,jev`, and travel with `--tool-format yaml`. The DeepSeek V4.1 Flash
run adds `--llm-config experiments/agentdojo/deepseek_llm_config.json` and `--workers 8` to every call, then runs the
same calls again with `--retry-errors`, which plays again any episode that ended in an error. Its manifest
is `experiments/agentdojo/manifest_deepseek-v4.1-flash.json`.

### Attack wordings

See `experiments/attack_wordings/README.md`. Jev only, no agent.

## Data rewrites

The shipped records were rewritten before release. Home directory paths, file owner names, account ids, labels, notes,
commit ids and private endpoint details were replaced, and account usage fields were dropped. Plays that are not
counted are left out of the data. Measured values were not changed. In the task data, six upstream fixture files named
`.env` or `types.env` were left out, and every task's `check.py` has a cleaned docstring, so `tasks.json` holds the
sha256 of the shipped task dirs.
`data/README.md` lists each change.
