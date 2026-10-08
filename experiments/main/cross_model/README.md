# Cross-model numbers

`xmodel.py` compares the three agent models on why the relevance filter cuts more tokens for one model than for
another. It measures how much raw tool output each model's plain runs pull in, how they read files, and the token
cut of each filter with its bootstrap interval. It is offline, makes no model or Jev call, and needs only `python3`
and its standard library.

```sh
uv run python scripts/unpack_data.py                   # once, from the repository root
python3 experiments/main/cross_model/xmodel.py         # writes out.json and per_task.csv beside it
python3 experiments/main/cross_model/xmodel.py --out <folder>
```

`../paper_tables.py` runs it in process, with its own output folder, and reads its `out.json` for the token cut in
Figure 3 and for Figure 4.

`reading_details.py` prints the reading and latency details the paper gives beside Figure 4, at 32K, using
`xmodel.py`'s call classes and the same run selection: per model, the clean Control runs with a whole-file read and
the share of raw characters from whole-file reads, the runs with 10 or more ranged reads and the median lines per
ranged read, relevance's clean outputs over 4,000 characters by class with the share of characters kept (all, and
reads only), and the median latency of a Jev request (`latency_ms` in `trace.jsonl`, send to parsed reply at the
client) over every relevance + injection run.

```sh
python3 experiments/main/cross_model/reading_details.py     # stdlib only; prints, writes nothing
```

## Inputs

- The three agent configs in `../analysis/configs/`: `gemma4-31b.json`, `mistral-small-3.2-24b.json` and
  `deepseek-v4.1-flash.json`. Each names its agent's 32K and 16K manifests.
- The run folders those manifests name, `experiments/runs-<tag>/main/`: the records, and each run's `trace.jsonl`
  (its tool calls and their output events).

It imports the analysis loader, `../analysis/data.py`, and selects runs the same way: per (configuration, task, rep)
the last record that is neither `rerun` nor `stopped`. A run that failed on every try is left out, and a paired
measure drops its pair. A 16K run that carries over is read from the 32K folder.

The token measure is the record's `total_prompt_tokens`: the agent's prompt tokens over every agent and compaction
call, with no Jev tokens.

## What it computes

Per agent model:

1. **Raw tool output of the plain runs** at 32K, on the clean versions and on all versions. Per run: the raw
   characters of read and bash output, before any filter or the cap; the number of reads and bash calls; whole-file
   reads against ranged reads; and bash calls by their first command word: search (`grep`, `rg`), slice (`sed`,
   `head`, `tail`, `awk`), dump (`cat`) or other. Means and medians per run, and the shares of each kind.
2. **Relevance against plain, clean pairs** at 32K: per task, the plain run's raw characters, both runs' tokens, the
   difference and the cut. The mean difference has a t interval.
3. **Spearman's rho** between a plain run's raw characters and the token difference, within each model and pooled
   over the three, with a two-sided permutation p. It also gives the share of the plain and relevance runs' outputs
   over 4,000 characters, which are the only ones the relevance filter chunks, and the share of their characters
   that relevance kept.
4. **The token cut** of each filter against plain, on clean pairs, at 32K and 16K: 1 - mean(filter) / mean(plain).
   It has a t interval and a bootstrap 95% interval that resamples tasks with their pairs.

The bootstrap and the permutation tests use 10,000 draws with seed 20261004.

## Outputs

`--out` names the folder, by default this one. Both files are not in git.

- `out.json`: one key per model (`gemma`, `mistral`, `deepseek`) with `counted_32k`, `failed_every_try_32k`,
  `counted_16k`, `plain_per_run` (item 1), `rel_clean` (item 2), `spearman_raw_vs_diff` and `gt4k` (item 3), and
  `ratio_cut` (item 4, by compaction threshold and filter). Then `pooled` (the pooled rho) and `method` (the token field, seed,
  draws and the 4,000-character pass-through).
- `per_task.csv`: one row per model and clean task, relevance against plain at 32K: `model`, `task`, `testbed`,
  `plain_raw_chars`, `plain_tokens`, `rel_tokens`, `diff`, `cut_pct`.

It also prints the numbers as tables.
