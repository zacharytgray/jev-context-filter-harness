# Versions

The exact versions and settings behind the shipped runs. Each value was checked against the run records where the
records hold it, not only against the code or the manifests. The last column names what confirms it. Paths are as
unpacked by `scripts/unpack_data.py`. The test-bed run sets are in `experiments/runs-<tag>/`, with the six tags
`gemma4-31b-32k`, `gemma4-31b-16k`, `mistral-small-3.2-24b-32k`, `mistral-small-3.2-24b-16k`,
`deepseek-v4.1-flash-32k` and `deepseek-v4.1-flash-16k`. The AgentDojo runs are in
`experiments/runs-agentdojo-gemma4-31b/` and `experiments/runs-agentdojo-deepseek-v4.1-flash/`.

Counts below are over the run dirs in each folder's `main/` tree: every run played in that set, the six Mistral runs
that failed on every try included. A 16K folder holds only the runs played at 16K; the runs it carries over are
counted in the 32K folder.

The prompts themselves are in [PROMPTS.md](PROMPTS.md).

## Agent models

| What | Value | Checked against |
|---|---|---|
| Gemma 4 31B | `google/gemma-4-31b-it` on OpenRouter (`https://openrouter.ai/api/v1`) | `model` in every completed call row of `llm.jsonl` (retry rows have none): 4,489 calls in `gemma4-31b-32k`, 2,721 in `gemma4-31b-16k`; `model` in all 858 records. The row's field is the model the response names |
| Gemma provider | Novita (bf16), pinned: `{"order": ["Novita"], "only": ["Novita"], "allow_fallbacks": false}` | `provider` in the same rows: Novita in every one |
| Gemma sampling | temperature 1.0, top_p 0.95, top_k 64 | `sampling` in all 858 records |
| DeepSeek V4.1 Flash | `deepseek/deepseek-v4.1-flash` on OpenRouter | `model` in every completed call row: 6,595 calls in `deepseek-v4.1-flash-32k`, 3,211 in `deepseek-v4.1-flash-16k`; `model` in all 798 records |
| DeepSeek provider | Morph (fp8), pinned the same way, no fallbacks | `provider` in the same rows: Morph in every one |
| DeepSeek sampling | temperature 1.0, top_p 0.95, reasoning effort low | `sampling` in all 798 records holds temperature and top_p. The effort is sent as `reasoning.effort`; the records do not hold it, so it is checked against the manifests' `reasoning_effort` and `s1harness/llm.py`. 8,784 of the call rows hold reasoning text, which the harness logs and never sends back |
| Mistral Small 3.2 24B | weights `mistralai/Mistral-Small-3.2-24B-Instruct-2506` at revision `95a6d26c4bfb886c58daf9d3f7332c857cb27b43`, served as `mistral-small-3.2-24b` | `model` in every completed call row: 5,762 calls in `mistral-small-3.2-24b-32k`, 2,620 in `mistral-small-3.2-24b-16k`; `model` in all 793 records. The revision is the server's `--revision`, in the Mistral manifests' `endpoint.serve_flags` |
| Mistral serving | a self-hosted vLLM 0.30.0 server on one 48 GB GPU (NVIDIA RTX 6000 Ada Generation), bf16 checkpoint quantized to FP8 on load (dynamic per-tensor), KV cache bf16 | the server's start-up log, which is not shipped; no `provider` in any call row |
| Mistral sampling | temperature 0.15, nothing else sent; vLLM's defaults hold for the rest | `sampling` in all 793 records |
| Max output | 8,192 tokens, every agent | `compaction.max_output` in every run's `levers.json` |
| Max turns | 40 (test beds) | `constants.max_turns` in every run's `levers.json` |
| AgentDojo agent, Gemma run | `google/gemma-4-31b-it` on Novita, temperature 1.0, top_p 0.95, top_k 64 | 7,507 calls in the Gemma run's `llm_calls.jsonl`; `sampling` in all 2,092 of its records, which also carry `provider_pin: Novita` |
| AgentDojo agent, DeepSeek run | `deepseek/deepseek-v4.1-flash` on Morph, temperature 1.0, top_p 0.95, no top_k, max_tokens 8,192, reasoning effort low | `model` and `provider` (Morph) in all 7,839 call rows of the DeepSeek run's `llm_calls.jsonl`; `sampling`, which here includes `reasoning_effort`, and `provider_pin: Morph` in every one of its records. 7,305 call rows hold reasoning text, which the runner logs and never sends back |
| AgentDojo turn limit | AgentDojo's own `ToolsExecutionLoop` default, 15 rounds | no Gemma episode reached it: every Gemma record has `end_state: answered`. In the DeepSeek run 17 episodes did (`end_state: max_iters`), all in slack: 1 stock and 16 filtered |

Every agent call and every compaction call counts in the rows above.

### The vLLM server

The Mistral server served `mistralai/Mistral-Small-3.2-24B-Instruct-2506` with these flags. Both Mistral
manifests keep them as `endpoint.serve_flags`:

```
--revision 95a6d26c4bfb886c58daf9d3f7332c857cb27b43 --served-model-name mistral-small-3.2-24b \
--host 127.0.0.1 --port 8000 \
--tokenizer-mode mistral --config-format mistral --load-format mistral \
--quantization fp8 --limit-mm-per-prompt '{"image":0}' \
--max-model-len 65536 --max-num-seqs 8 --gpu-memory-utilization 0.92 \
--enable-prefix-caching --enable-prompt-tokens-details \
--enable-auto-tool-choice --tool-call-parser mistral --generation-config vllm
```

Packages in the server's environment: vllm 0.30.0, transformers 5.16.1, mistral-common 1.12.0, torch 2.13.0,
flashinfer-python 0.6.18.post1, triton 3.7.1. transformers is pinned to 5.16.1 because vLLM 0.30.0's Pixtral loader
imports a class that later transformers releases no longer define. With images set to 0 the model runs text only.
`--generation-config vllm` keeps the checkpoint's generation config from setting server defaults. There was no
speculative decoding. These server details come from the server's log and environment, which are not shipped.
The prompt and completion token counts in `llm.jsonl` are vLLM's own usage counts.

## Jev

| What | Value | Checked against |
|---|---|---|
| Endpoint | `POST https://api.typesafe.ai/v1/systemone` | `s1harness/jev.py` (`URL`); the records hold the body, not the URL |
| Model name sent | `jev-latest` | `body.model` in every Jev request row of every `trace.jsonl`, and of AgentDojo's `jev_trace.jsonl`; `constants.jev_model` in every `levers.json` |
| Version served | `jev-1.13.0` | `model` in every Jev response: 67,852 requests in `gemma4-31b-32k`, 13,928 in `gemma4-31b-16k`, 51,634 in `mistral-small-3.2-24b-32k`, 8,694 in `mistral-small-3.2-24b-16k`, 12,328 in `deepseek-v4.1-flash-32k`, 5,390 in `deepseek-v4.1-flash-16k`, 8,993 in the Gemma AgentDojo run, 14,031 in the DeepSeek AgentDojo run, 11,946 in the attack-wordings check. No other value occurs |

## Compaction

| What | Value | Checked against |
|---|---|---|
| Context window | 262,144 tokens for Gemma and DeepSeek (the harness default; DeepSeek's endpoint lists 1,048,576), 65,536 for Mistral (the server's `--max-model-len`, passed as `--context-window`) | `compaction.context_window` in every `levers.json` |
| Margin | 4,096 tokens | `compaction.margin` in every `levers.json` |
| Compaction threshold, 32K sets | 32,768 tokens | `compaction.budget` in all 600 runs of each 32K folder |
| Compaction threshold, 16K sets | 16,384 tokens for the runs played at 16K (258 Gemma, 193 Mistral, 198 DeepSeek); 32,768 for the runs carried over from 32K, whose prompt never passed 16,384 before their final call | `compaction.budget` per run; the carry-over rule is `choice` in each 16K manifest |

## Run dates

Dates as the records give them (`started`, local time, UTC-5).

| Run | First | Last |
|---|---|---|
| `gemma4-31b-32k` | 2026-10-02 | 2026-10-02 |
| `gemma4-31b-16k` | 2026-10-02 | 2026-10-02 |
| `mistral-small-3.2-24b-32k` | 2026-10-02 | 2026-10-03 |
| `mistral-small-3.2-24b-16k` | 2026-10-04 | 2026-10-04 |
| `deepseek-v4.1-flash-32k` | 2026-10-02 | 2026-10-03 |
| `deepseek-v4.1-flash-16k` | 2026-10-03 | 2026-10-03 |
| AgentDojo, Gemma | 2026-09-30 | 2026-09-30 |
| AgentDojo, DeepSeek | 2026-10-04 | 2026-10-05 |
| Attack-wordings check | 2026-10-01 | 2026-10-01 |

## AgentDojo

| What | Value | Checked against |
|---|---|---|
| Package | `agentdojo` 0.1.35 | `agentdojo_package_version` in all 2,092 episode logs of each run |
| Source commit | `089ed468cf3ed0322acc66b0211f26d9d90dbf60` | `commit.agentdojo_commit` (`089ed468`) in every record of both runs |
| Benchmark version | v1.2.2 | `benchmark_version` in every record and every episode log |
| System prompt | AgentDojo's default, `load_system_message(None)` | all 2,092 episode logs of each run open with the same system message, the same in both runs |
| Attack | `important_instructions`, user "Emma Johnson", model "AI assistant" | `attack_type` in the logs: 1,898 attacked episodes, 194 with no attack, in each run |
| Pipeline | `SystemMessage`, `InitQuery`, the LLM, then `ToolsExecutionLoop` over the tool executor, the defense and the LLM; pipeline name `openai-compatible_gemma-4-31b-it_<defense>` and `openai-compatible_deepseek-v4.1-flash_<defense>` | `pipeline_name` in the logs; `build_pipeline` in `experiments/agentdojo/run_agentdojo.py` |
| Tool output format | AgentDojo's `tool_result_to_str` for banking, slack and workspace; `--tool-format yaml` for travel | `tool_format` in the records: `yaml` in every travel record; in the other suites `str`, or no field in the Gemma run, where the runner's default `str` applied |
| Departures from stock | four in both runs: the sampling above, the OpenRouter provider pin, travel as YAML, and an error result instead of an exception for a tool call with invalid JSON arguments (`ArgCheckExecutor`) | `build_pipeline` in `experiments/agentdojo/run_agentdojo.py`; `deviations_from_stock_agentdojo` in `manifest_deepseek-v4.1-flash.json` |
| Tool-call id replacement | both runners give a fresh id to an empty or repeated tool-call id (the Gemma run's runner as `call_<n>_<j>`, the DeepSeek run's as `fresh_id`, 9 letters and digits). It never triggered: all 7,573 Gemma and 12,003 DeepSeek tool-call ids are the model's own, none empty or repeated | `fresh_id` in `experiments/agentdojo/run_agentdojo.py`; `tool_calls[].id` in each suite's `llm_calls.jsonl` (logged after any replacement) |

## Software

| What | Value | Checked against |
|---|---|---|
| Harness Python | 3.12 or later | `requires-python` in `pyproject.toml` and `uv.lock` |
| Python packages | openai 3.19.1, httpx 0.28.1, python-dotenv 1.2.3, pytest 9.1.1 | `uv.lock`. The run records do not name package versions |
| uv | not recorded | `uv.lock` is lock format version 1, revision 3; no record names the uv version |
| Task test environment | Python 3.14 and pytest 9.1.1, in a separate venv first on the agent's `PATH` (`experiments/run.py`, `task_env`) | `task_env: true` in every record of the five complete test-bed sets |
| Operating system | macOS, for the harness and the agents' workdirs | the agent's working directories in the transcripts are under `/private/tmp/s1h-eval/`, `experiments/run.py`'s macOS default. The vLLM server ran on Linux |

## Datasets

The upstream dataset revisions and licenses are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
