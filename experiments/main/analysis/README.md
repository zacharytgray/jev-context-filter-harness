# Analysis loader

`data.py` reads one agent model's run records and computes every measure of the test-bed experiment for it: the four
configurations on the same 150 task versions at the 32K and the 16K compaction threshold. It is offline, reads only the run folders
and the manifests, makes no model or Jev call, and needs only `python3` and its standard library.
`../cross_model/xmodel.py`, `../cost_ratio.py` and `../paper_tables.py` import it, so they all count the same runs.
`../paper_tables.py` builds the paper's Table 2 and Figure 3 from its measures.

## Command

From the repository root, after `uv run python scripts/unpack_data.py`:

```sh
python3 experiments/main/analysis/data.py --config experiments/main/analysis/configs/gemma4-31b.json
python3 experiments/main/analysis/data.py --config experiments/main/analysis/configs/mistral-small-3.2-24b.json
python3 experiments/main/analysis/data.py --config experiments/main/analysis/configs/deepseek-v4.1-flash.json
```

Each call writes `out-<agent>/data.json` next to `data.py` (not in git). Flags: `--config` (the agent config; default
`configs/gemma4-31b.json`), `--root` (the repository root the run sets are under; default this repository),
`--b16k-manifest` (default: the config's 16K manifest) and `--out` (the output file).

## Configs

One config per agent in `configs/`. Each names the model, its 32K and 16K manifests and their run tags, and the
expected run counts. Nothing else. Paths are relative to the repository root.

| Config | Agent | Run sets |
|---|---|---|
| `gemma4-31b.json` | Gemma 4 31B | `gemma4-31b-32k`, `gemma4-31b-16k` |
| `mistral-small-3.2-24b.json` | Mistral Small 3.2 24B | `mistral-small-3.2-24b-32k`, `mistral-small-3.2-24b-16k` |
| `deepseek-v4.1-flash.json` | DeepSeek V4.1 Flash | `deepseek-v4.1-flash-32k`, `deepseek-v4.1-flash-16k` |

## What it counts

Set A is the 32K manifest: its played runs under `experiments/runs-<tag>/main/`. Set B is the 16K manifest: its runs
played at 16K under its own runs folder, and its `reused` runs, which carry over from the 32K folder.

- Per (configuration, task, rep) the counted record is the last one that is neither `rerun` nor `stopped`.
- A run whose counted record ended in an error, a context overflow, a crash or a timeout failed on every try. It is
  left out of every measure and listed under `final`. A paired measure drops its pair.
- A run that reached the 40-turn limit is a finished run and is counted.

## Output keys

`data.json` holds the 32K measures at the top level and the same keys for the 16K set under `k16`. Rows are per test
bed (`contextbench`, `legalbenchrag`, `docfinqa`) and pooled (`all`), per configuration (`plain`, `relevance`,
`injection`, `relevance-injection`).

- `runs`: the number of counted runs.
- `score`: the clean score per configuration and test bed: mean with its t interval, n, and the runs that scored 0.
  The score over all versions, which the paper reports, is computed by `../paper_tables.py` (`score_abs.csv`).
- `tokens`: per configuration and test bed, over the clean runs: the median total prompt tokens (`prompt`),
  compactions, outputs cut at the 50,000-character cap (`capped`), the median wall time, and the runs that reached
  the 40-turn limit.
- `perrun`: per configuration and test bed, over the clean runs: wall time and total prompt tokens per run, mean with
  its t interval and median.
- `config`: per configuration, over all its runs, clean and planted: the run count, compactions (`comps`) and the runs
  that compacted (`comp_runs`), the wall median, Jev input tokens (`jev_tok`), Jev requests, Jev errors and fallbacks,
  and the runs that reached the 40-turn limit (`maxturns`).
- `rates`: per configuration and test bed, over the planted runs: attacks followed (`harm`) and planted text reached
  (`reached`), each k of n with its Wilson interval. For each filter, `d_harm` and `d_reached` are the paired
  difference from plain with Newcombe's paired interval, and the tasks that changed each way.
- `planted`: per configuration, pooled over the planted runs: attacks followed, planted text reached and the mean
  planted score.
- `util`: per configuration and test bed: the mean planted score, and the planted minus clean score per task in the
  same configuration (`minus_clean`), mean with its t interval.
- `paired`: each filter against plain, per test bed and pooled: the clean score difference per task, mean with its t
  interval. The difference over all versions is computed by `../paper_tables.py` (`score_versions.csv`).
- `pwall`: each filter against plain: the wall time difference per task version, mean with its t interval, over all
  versions (`v` = `both`) and over the clean versions (`v` = `clean`).
- `ptok`: each filter against plain, per test bed and over clean, planted and all versions: the per-task cut
  1 - filter / plain in total prompt tokens (mean with its t interval, and median), and the mean token difference.
- `final`: the runs that failed on every try, with their configuration, task, end and error.
- `k16`: the same keys for the 16K set.

The token cut in mean tokens, 1 - mean(filter) / mean(plain), with its bootstrap interval, is computed by
`../cross_model/xmodel.py`, not here.

## Intervals

All intervals are 95%.

- A mean, or a mean paired difference, gets a Student-t interval: mean plus or minus t(0.975, n - 1) sd / sqrt(n).
- A single rate gets a Wilson interval.
- A paired rate gets Newcombe's paired score interval (method 10), with his adjustment to phi.

## AgentDojo

`agentdojo(root, folders, arms)` reads the AgentDojo run folders and returns AgentDojo's three rates: attack success,
utility under attack and utility without attack. It pairs the stock agent (defense `none`) with the injection filter
(defense `jev-injection`) case by case, per suite and overall. Per record key the last record stands, and a `rerun`
record is not a result. Each rate has its Wilson interval per arm, and the filter minus stock difference has
Newcombe's paired interval. The default root is `experiments/runs-agentdojo-gemma4-31b`. `main()` does not call it.
`../paper_tables.py` calls it on `experiments/runs-agentdojo-gemma4-31b` and
`experiments/runs-agentdojo-deepseek-v4.1-flash` for our rows of Table 3. See `experiments/agentdojo/RESULTS.md`.
