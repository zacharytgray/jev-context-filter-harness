# Test-bed experiment

The four configurations (plain, relevance, injection, relevance + injection) on 150 task versions: 25 tasks per test
bed (ContextBench, LegalBench-RAG, DocFinQA), each in a clean and a planted version. The paper calls plain Control.
Three agent models, each with one run per task version and configuration at a 32,768-token compaction threshold (32K)
and at 16,384 (16K). The results are in [RESULTS.md](RESULTS.md).

## Run sets

| Run set (tag) | Agent | Threshold | Runs played | Carried from 32K | Manifest |
|---|---|---|---|---|---|
| `gemma4-31b-32k` | Gemma 4 31B | 32,768 | 600 | | `manifest_gemma4-31b-32k.json` |
| `gemma4-31b-16k` | Gemma 4 31B | 16,384 | 258 | 342 | `manifest_gemma4-31b-16k.json` |
| `mistral-small-3.2-24b-32k` | Mistral Small 3.2 24B | 32,768 | 600 (6 failed on every try) | | `manifest_mistral-small-3.2-24b-32k.json` |
| `mistral-small-3.2-24b-16k` | Mistral Small 3.2 24B | 16,384 | 193 (1 failed on every try) | 407 (5 as failures) | `manifest_mistral-small-3.2-24b-16k.json` |
| `deepseek-v4.1-flash-32k` | DeepSeek V4.1 Flash | 32,768 | 600 | | `manifest_deepseek-v4.1-flash-32k.json` |
| `deepseek-v4.1-flash-16k` | DeepSeek V4.1 Flash | 16,384 | 198 | 402 | `manifest_deepseek-v4.1-flash-16k.json` |

Each 32K set has all 600 runs of its agent, none taken from another set. Each 16K set is its 32K set with the runs
played at 16K in place of their 32K runs (below).

## Files

| file | what it is |
|---|---|
| `test.toml` | the test: maps the four configurations to their lever files in `levers/` |
| `tasks.json` | the 150 task versions: the draw (`selection`), then per task its id, test bed, variant, sha256 and size facts |
| `manifest_<tag>.json` | the frozen run manifest of each run set: model, endpoint or provider, sampling, context window, compaction threshold (`budget`), the runs to play and, for a 16K set, the runs it carries over (`reused`) and the 32K failures it carries (`failed_32k`) |
| `RESULTS.md` | the results |
| `analysis/data.py` | the analysis loader: the counted runs of one agent and every measure over them, written to `data.json`; one config per agent in `analysis/configs/` ([README](analysis/README.md)) |
| `cross_model/xmodel.py` | the cross-model script: raw tool output per model and the token cut with its bootstrap interval ([README](cross_model/README.md)) |
| `cost_ratio.py` | Jev input tokens per agent prompt token saved, with bootstrap intervals; writes `cost_ratio.csv` (needs numpy) |
| `compaction.py` | compaction against Control and 16K against 32K: share of runs that compacted (Newcombe's paired interval) and compactions per run (t interval), paired by task version |
| `paper_tables.py` | rebuilds the paper's Table 2 and the data behind its figures and cost numbers (needs numpy) |
| `tables/` | `comparison.md` and `comparison.tex` (the paper's Table 3: our AgentDojo runs next to published defenses), `references.bib` |
| `tasks/` | links to the task dirs, made by `scripts/unpack_data.py` (not in git) |

The run records are in `experiments/runs-<tag>/main/<configuration>/`. Each `records.jsonl` has one line per run
played in that set. A 16K folder holds only the runs played at 16K; the carried runs stay in the 32K folder, and the
16K manifest's `reused` list names each one. The per-run dirs (`<configuration>/<task>/r0/`) come from the tarballs in
`data/`. Each run set ships only its counted plays.

## How the runs were launched

`experiments/driver.py` reads a manifest and starts `experiments/run.py` with the manifest's settings. It logs the
block's progress to the JSON lines file given as its second argument. It has two modes.

- **OpenRouter mode** (Gemma, DeepSeek). The driver watches the OpenRouter key's usage and the Jev tokens in the runs'
  traces. It stops every run if either passes the manifest's `stop_at`. It reads `OPENROUTER_API_KEY` from the
  environment, not from `.env`, so export it first (`set -a; . ./.env; set +a`).
- **Local endpoint mode** (Mistral). A manifest with `endpoint: {"kind": "local", "base_url": ..., "api_key_env":
  null}` and no provider runs on a self-hosted server. There is no OpenRouter key; the Jev stop works as before. The
  driver refuses to launch unless `GET <base_url>/models` lists the manifest's model, and it passes the manifest's
  `context_window_tokens` as `--context-window`.

```sh
uv run python experiments/driver.py experiments/main/manifest_gemma4-31b-32k.json <log>.jsonl
uv run python experiments/driver.py experiments/main/manifest_gemma4-31b-32k.json --dry-run    # lists the runs left to play, no key, no call
```

The driver starts `run.py` with the repository's own `.venv/bin/python`, so run `uv sync` first.

For the Gemma 32K manifest the driver ran this, with 12 runs at once:

```sh
uv run python experiments/run.py main --arms plain,relevance,injection,relevance-injection \
    --tasks <the manifest's task list> --start 0 --repeats 1 --workers 12 --tag gemma4-31b-32k \
    --model google/gemma-4-31b-it --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
    --provider Novita --max-output 8192 --temperature 1.0 --top-p 0.95 --top-k 64 \
    --max-turns 40 --compaction-budget 32768 --skip-done --note "<the manifest's note>"
```

For DeepSeek the model is `deepseek/deepseek-v4.1-flash`, the provider `Morph`, and the sampling `--temperature 1.0
--top-p 0.95 --reasoning-effort low`, with 12 runs at once. For Mistral, with 8 runs at once (the server's
`--max-num-seqs 8`):

```sh
uv run python experiments/run.py main --arms plain,relevance,injection,relevance-injection \
    --tasks <the manifest's task list> --start 0 --repeats 1 --workers 8 --tag mistral-small-3.2-24b-32k \
    --model mistral-small-3.2-24b --base-url http://127.0.0.1:8000/v1 --max-output 8192 --context-window 65536 \
    --temperature 0.15 --max-turns 40 --compaction-budget 32768 --skip-done --note "<the manifest's note>"
```

The base URL is wherever the vLLM server listens; the manifest records `http://127.0.0.1:18000/v1`, where the
server was reached. The server needs no key. Its launch flags are in
[docs/VERSIONS.md](../../docs/VERSIONS.md).

`run.py` copies each task's `repo/` into a fresh git workdir under `/private/tmp/s1h-eval/<tag>/` (`/tmp/s1h-eval/`
off macOS; `S1H_WORK_ROOT` moves it), runs the harness CLI on the task's request, then runs the task's `check.py`,
which scores the run and checks for the attack's receipt. It appends one record per run to
`<configuration>/records.jsonl`. It plays a task's four configurations next to each other, and takes ContextBench,
LegalBench-RAG and DocFinQA tasks in turn, so a partial stop leaves paired data. Each run gets 3,600 s; then it is
sent SIGTERM and writes its records. A record is marked `dirty` when the harness, the lever files, the runner or the
test folder had uncommitted changes. No counted record in any run set is dirty.

## Failed runs

A run that ends in an error is played again, up to four tries, and the last play stands. `run.py --skip-done` plays
only the runs with no record. A run that reaches 40 turns is finished, not failed, and is kept. A provider 429 that
the harness retried and got past is not a failure.

Six Mistral runs per compaction threshold failed on every try; we leave them and their pairs out of every measure. Their last play
stays in `records.jsonl` with `end: error`. At 16K, five of them are the 32K failures carried over (below). No Gemma or
DeepSeek run is left out.

## The 16K step

Each 32K run's `llm.jsonl` is read. A run whose prompt stayed under 16,384 tokens before its final call never compacts
at either compaction threshold, so its 32K run also counts at 16K. Such a run carries over: it is its own 16K run, and the 16K
manifest lists it in `reused`. Every other run is played again at `--compaction-budget 16384`. Every other setting of
a 16K manifest equals its 32K one. Each 16K manifest records the rule (`choice`) and the run it carries for each task
(`reused`).

- Gemma: 342 runs carried, 258 played at 16K.
- DeepSeek: 402 carried, 198 played.
- Mistral: 407 carried and 193 played. Five of the runs that failed on every try at 32K stayed under 16,384 tokens
  before their final call, so they carry over as failures; the manifest lists them in `failed_32k`. They stay out of
  every 16K measure.

So two compaction thresholds share most of their runs, and they are not independent checks.

## How the runs were analysed

All analysis is offline and reads only the run folders and the manifests. Unpack the data first:

```sh
uv run python scripts/unpack_data.py
```

The analysis loader, `analysis/data.py`, takes one agent's config, reads its 32K and 16K manifests, and takes per
(configuration, task, rep) the last record that is neither `rerun` nor `stopped`. A run that failed on every try is
listed as failed and left out of every measure; a paired measure drops its pair. A 16K run that carries over is read
from the 32K folder.

To rebuild the paper's Table 2 and the data behind its figures in one step:

```sh
uv run --with numpy python experiments/main/paper_tables.py              # writes to experiments/main/out/
uv run --with numpy python experiments/main/paper_tables.py --out <folder>
```

It writes `t1_main.csv` and `t1_main.tex` (Table 2), `f2_effects.csv` (Figure 3), `f3_reading.csv`, `f3_tasks.csv`
and `f3_rho.csv` (Figure 4), `cost_ratio.csv` (the cost ratios and break-even prices of Section 5.7) and `t4_agentdojo.csv` (our two rows of Table 3).
It also writes `score_abs.csv` and `score_versions.csv`: the scores and the score differences from Control over all
versions, the paper's score measure, and over the clean and the planted versions. It writes `xmodel.py`'s `out.json`
and `per_task.csv` to `cross_model/` inside that folder. It writes CSV and JSON only, no plots. It runs the analysis
loader, `xmodel.py` and `cost_ratio.py` in process and takes about a minute. The output folder is not in git.

The same steps one at a time:

```sh
python3 experiments/main/analysis/data.py --config experiments/main/analysis/configs/gemma4-31b.json
python3 experiments/main/analysis/data.py --config experiments/main/analysis/configs/mistral-small-3.2-24b.json
python3 experiments/main/analysis/data.py --config experiments/main/analysis/configs/deepseek-v4.1-flash.json
python3 experiments/main/cross_model/xmodel.py
uv run --with numpy python experiments/main/cost_ratio.py
```

Each `data.py` call writes `experiments/main/analysis/out-<agent>/data.json`, which is not in git. See
[analysis/README.md](analysis/README.md). `xmodel.py` reads the three 32K sets (and the 16K sets) through the
analysis loader, so it counts the same runs, and writes `out.json` and `per_task.csv` beside it
([cross_model/README.md](cross_model/README.md)). `cost_ratio.py` reads the same runs through the analysis loader and
writes `cost_ratio.csv` beside it. It needs numpy, which is not a dependency of the project, hence `--with numpy`.

`experiments/run_card.py` shows one run's tool calls, what each filter did to each output, and where its prompt tokens
went. `--rollup <runs folder>` gives a table per configuration and task.
