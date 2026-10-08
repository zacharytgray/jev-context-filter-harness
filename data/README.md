# Data

The run records and the task data, as zstd-compressed tarballs. Each file is under 95 MB. `SHA256SUMS` lists the
sha256 of every tarball.

| tarball | contents | unpacks to |
|---|---|---|
| `runs-gemma4-31b-32k.tar.zst` | Gemma 4 31B at 32K: 600 runs, 150 per configuration | `experiments/runs-gemma4-31b-32k/main/<configuration>/<task>/r0/` |
| `runs-gemma4-31b-16k.tar.zst` | Gemma 4 31B at 16K: the 258 runs played at 16K | `experiments/runs-gemma4-31b-16k/main/<configuration>/<task>/r0/` |
| `runs-mistral-small-3.2-24b-32k.tar.zst` | Mistral Small 3.2 24B at 32K: 600 runs | `experiments/runs-mistral-small-3.2-24b-32k/` |
| `runs-mistral-small-3.2-24b-16k.tar.zst` | Mistral Small 3.2 24B at 16K: the 193 runs played at 16K | `experiments/runs-mistral-small-3.2-24b-16k/` |
| `runs-deepseek-v4.1-flash-32k.tar.zst` | DeepSeek V4.1 Flash at 32K: 600 runs | `experiments/runs-deepseek-v4.1-flash-32k/` |
| `runs-deepseek-v4.1-flash-16k.tar.zst` | DeepSeek V4.1 Flash at 16K: the 198 runs played at 16K | `experiments/runs-deepseek-v4.1-flash-16k/` |
| `agentdojo-gemma4-31b.tar.zst` | the AgentDojo run with Gemma 4 31B: both defenses, all four suites (travel in `travel-yaml/`) | `experiments/runs-agentdojo-gemma4-31b/`: one folder per suite |
| `agentdojo-deepseek-v4.1-flash.tar.zst` | the AgentDojo run with DeepSeek V4.1 Flash, laid out the same way | `experiments/runs-agentdojo-deepseek-v4.1-flash/`: one folder per suite |
| `attack-wordings.tar.zst` | the attack-wordings check: its items and the run's results, Jev trace, Jev token log (`spend.jsonl`), LLM log and run log | `experiments/attack_wordings/items.jsonl` and `experiments/attack_wordings/run/` |
| `tasks-contextbench-*.tar.zst` | the 50 ContextBench task dirs (25 clean, 25 planted), split into parts by size | `data/tasks/contextbench/<task>/` |
| `tasks-legalbenchrag.tar.zst` | the 50 LegalBench-RAG task dirs | `data/tasks/legalbenchrag/<task>/` |
| `tasks-docfinqa.tar.zst` | the 50 DocFinQA task dirs | `data/tasks/docfinqa/<task>/` |

ContextBench's task dirs are split into three tarballs, `tasks-contextbench-1.tar.zst` to `-3`, to keep each under
95 MB. Every other set is one tarball.

A 16K run set holds only the runs played at 16K. The runs it carries over from 32K stay in the 32K folder, and the
16K manifest (`experiments/main/manifest_<tag>.json`, key `reused`) names each one. The analysis reads them from
there.

The `records.jsonl` files of the six test-bed run sets are in git, uncompressed, at
`experiments/runs-<tag>/main/<configuration>/records.jsonl`. The analysis loader
`experiments/main/analysis/data.py` and the cross-model script read the records and the per-run `llm.jsonl` and
`trace.jsonl` files. The attack-wordings report is in git too, at `experiments/attack_wordings/run/report.json`.

## Notes on the runs

- **Counted plays only.** The data holds one play per run and one per AgentDojo episode: the play the analysis
  counts. Plays that are not counted are left out.
- **Failed runs.** Six Mistral Small 3.2 24B runs per compaction threshold failed on every try. Their failing play is in the data,
  and we leave them and their pairs out of every measure.
- **Harness version.** The Gemma 4 31B AgentDojo run and the attack-wordings check used an earlier version of this
  harness with the same injection filter and chunk limits as the current one.

## Unpack and check

```sh
uv run python scripts/unpack_data.py
```

It needs Python 3.14 or later (its standard library reads zstd) or the `zstd` command on the PATH. It checks every
tarball against `SHA256SUMS`, unpacks the run dirs in place and the task dirs to `data/tasks/`,
checks each task dir's sha256 against `experiments/main/tasks.json`, and links the tasks into `experiments/main/tasks/`
for `experiments/run.py`. A task dir's sha256 is taken over the sorted list of (relative path, sha256 of the file's
bytes) for every file in the dir, with `__pycache__` and `.pyc` left out (`experiments/bench/common.py`, `task_hash`).

To check the tarballs by hand:

```sh
cd data && shasum -a 256 -c SHA256SUMS
```

The tarballs were packed with a 1 GB zstd window. To unpack one by hand, from the repository root:

```sh
zstd --long=30 -dc data/<name>.tar.zst | tar -x
```

## What a run dir holds

A test-bed run dir (`<configuration>/<task>/r0/`):

- `request.txt`: the request the agent got.
- `stderr.txt`: the harness CLI's error output.
- `work.diff`: what the agent changed in its workdir.
- `run/<timestamp>/`: `levers.json` (the configuration and constants), `tools.json` (the tool schemas the agent
  saw), `transcript.json` (every message), `llm.jsonl` (every LLM call with its token usage, provider and timing;
  DeepSeek's rows also hold its reasoning text), `trace.jsonl` (every tool output, every Jev request and filter
  decision), `result.json` (how the run ended), and `bench_score.json` (the scorer's full detail).

An AgentDojo suite folder holds `records.jsonl`, `defense.jsonl`, `jev_trace.jsonl`, `llm_calls.jsonl` and
`agentdojo/` (AgentDojo's own per-episode logs). `experiments/agentdojo/README.md` describes each.

A task dir holds `task.toml`, `repo/` (the workdir), `gold.json`, `check.py` and `bench_score.py`.
`experiments/bench/README.md` describes each.

## Changes made before release

The run data and the task data were changed in these ways before release. Measured values (scores, token counts,
timings, costs, Jev scores, filter decisions, Jev request ids, provider response ids, provider names and the `dirty`
flag) are unchanged.

Run data:

- Plays that are not counted are left out of the data, and so are the runners' own logs.
- Home directory paths are replaced with `/home/user/`, in records, transcripts, traces, LLM logs and AgentDojo logs.
- The file owner name in `ls -l` output the agent saw is replaced with `user`.
- In 26 DeepSeek runs the agent's `bash` listed or printed folders outside its workdir: the home folder, folders in
  it, or `/tmp`. The entry names in those listings, and private folder names in home paths, are replaced with
  placeholders (`dir_01`, `file_01` and so on), numbered within each run, wherever they appear in that run's files.
  Names of public tools and packages are kept.
- Provider account ids inside error messages in `llm.jsonl` are replaced with `user_REDACTED`. In the DeepSeek
  AgentDojo run the same ids, inside provider error messages in `records.jsonl` and `llm_calls.jsonl`, are replaced
  the same way.
- In the DeepSeek AgentDojo run, the paths of the code checkout it played from and of the AgentDojo install are
  replaced with `/home/user/jev-context-filter-harness`, and the run folder's path with its name here.
- The launcher output that held the API key's account usage is left out.
- Labels are renamed to the names used in this repository: the record's `test`, `run_dir` and `note`, the manifests'
  `suite`, `tag`, `block` and `note`, the agent workdir paths, the `relevance_wording` label in `levers.json` (the
  wording itself is unchanged), and the attack-wordings item keys that name the AgentDojo run (`agentdojo_*`).
- Each record's `commit` is set to null. It pointed to development history that is not public.
- The test-bed manifests keep only the fields that set up and describe the runs: block, date, tag, note, suite,
  model, provider or endpoint, max output, sampling, reasoning effort, context window, compaction threshold, turn limit,
  Jev model, `skip_done`, the task data, the 16K choice, cells, runs, `reused`, `budget` (the compaction threshold), `stop_at`
  and `reruns`. The `reruns` text states the rule: a failed run is played again, up to four tries, the last play
  stands, and a run that fails every try is reported failed and left out with its pairs. The Mistral 16K manifest
  also keeps `failed_32k`. Every other field is dropped: approvals, plans and next steps, launch and report commands, cost
  estimates, prices, the reasons written beside each setting, `measures`, and `stop_at.why`. For
  the self-hosted Mistral runs this also drops the server block (its host, GPU and serving settings), the tunnel
  command and the endpoint checks made before launch; the vLLM version and flags are in `docs/VERSIONS.md`. The
  kept `endpoint` gets a `note`. In the AgentDojo and attack-wordings manifests, fields that recorded approvals,
  account usage, people, or hashes of files from before these changes are dropped, and so are their cost estimates,
  spend stops and `measures`. The DeepSeek AgentDojo manifest also drops its commit, its checkout and key file paths,
  the trial run made before it and its provider comparison; its notes are rewritten as plain facts. The attack-wordings manifest's
  `items.sha256` is that of the shipped `items.jsonl`.

The analysis gives the same numbers on the changed data as on the original data.

Task data:

- Six upstream fixture files named `.env` or `types.env` are left out of the task tarballs: `repo/docs/.env` in the
  two versions of one material-ui task (`cb-mui-material-ui-28813-clean`, `cb-mui-material-ui-28813-attack-needed`),
  and `repo/test/integration/targets/config/files/types.env` in the two versions of two ansible tasks
  (`cb-instance-ansible-ansible-942424e10b2095a173dbd78e7128f52f7995849b-v30a923fb5c164d6cd18280c02422f75e611e8fb2-clean`
  and `-attack-needed`, `cb-instance-ansible-ansible-e9e6001263f51103e96e58ad382660df0f3d0e39-v30a923fb5c164d6cd18280c02422f75e611e8fb2-clean`
  and `-attack-needed`). These six task dirs differ from the agent's workdir in the runs by that one file each.
- Every task dir's `check.py` is the shipped `experiments/bench/check.py`, whose docstring was cleaned. Its code is
  unchanged.
- So the sha256 values in `experiments/main/tasks.json` are those of the shipped task dirs, not of the dirs used in the
  runs. `scripts/unpack_data.py` checks the shipped dirs against them.

## License

The task specifications, gold files, planted text and run records we added are CC BY 4.0. Upstream content inside the
task dirs and the records keeps its own license. See `THIRD_PARTY_NOTICES.md`.
