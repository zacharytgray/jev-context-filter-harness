# Third-party notices

This repository redistributes content from the sources below, in the task tarballs under `data/`, in the run records
(which quote tool output the agent read), and as copied text in `experiments/bench/attacks.py`. Each source keeps its
own license. This file lists each source, the version we used, its license, and the conditions that apply. The full
license texts are in `LICENSES/`, and `LICENSES/README.md` says where each text came from.

## Our own content

- **Code** in this repository: MIT, see `LICENSE`.
- **Data we added**: task specifications (`task.toml` requests and labels), gold answer files in the format our
  scorers read (`gold.json`), the planted text, `tasks.json`, the manifests, and the run records (records, transcripts,
  traces, LLM and Jev logs, AgentDojo episode logs): CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/, full
  text in `LICENSES/CC-BY-4.0.txt`).
- Upstream content inside these files (repository code, contract text, filings, benchmark questions and gold spans,
  AgentDojo environment data) keeps its upstream license, listed below.

## Modified files

Every planted task version (`<id>-attack-needed`) is a modified copy of its clean version. One file in its `repo/` has
AgentDojo's `important_instructions` text, with our goal, inserted where the task's answer is: as comment lines in a
source file (ContextBench), or as a plain paragraph in a contract (LegalBench-RAG) or a filing (DocFinQA). The labels
in `task.toml` record the file, the line or character span, and the full inserted text. Nothing else in the file is
changed. The inserted text is a test attack. It asks the agent to write a harmless receipt file.

Clean versions also differ from upstream in documented ways: ContextBench checkouts have no `.git` and symlinks are
resolved or dropped; LegalBench-RAG documents are renamed to safe file names; DocFinQA filings have tabs replaced by
spaces. `experiments/bench/README.md` gives each rule.

## ContextBench

- Source: Hugging Face dataset `Contextbench/ContextBench`, revision `c2855792b006af41c67202d33883fb9d46362853`;
  code https://github.com/EuniAI/ContextBench at `1436c28a8eb95496da4ea69ad458b9f8a8eb7d61`.
- Paper: Li et al., ContextBench, arXiv:2602.05892.
- License: Apache-2.0, text in `LICENSES/Apache-2.0-ContextBench.txt`. Apache-2.0 requires a copy of the license
  and a statement of changes. Our changes are listed
  under "Modified files" above and in `experiments/bench/README.md`. The ContextBench context request text in our
  requests is adapted from `agent-frameworks/mini-swe-agent/multi-poly-pro-verified/configs/swebench_context_aware.yaml`.
- ContextBench's tasks come from four upstream benchmarks:
  - SWE-bench Verified (arXiv:2310.06770): code MIT.
  - Multi-SWE-bench (arXiv:2504.02605): CC0, subject to the licenses of the included repositories.
  - SWE-PolyBench (arXiv:2504.08703): MIT.
  - SWE-bench Pro (arXiv:2509.16941): harness MIT, content under the licenses of the included repositories.

### Repository checkouts

Each ContextBench task dir holds a checkout of an open source repository at the task's base commit. Each checkout
keeps its own license file in its `repo/` folder, and that license applies to its content.

| Repository | Tasks | Upstream benchmark | License |
|---|---|---|---|
| clap-rs/clap | 2895, 3420 | Multi-SWE-bench | MIT or Apache-2.0 (dual; both files in the checkout) |
| cli/cli | 2250, 3608 | Multi-SWE-bench | MIT |
| django/django | 13028, 14999 | SWE-bench Verified | BSD-3-Clause |
| facebook/zstd | 1105, 1532 | Multi-SWE-bench | BSD or GPLv2 (dual) |
| fasterxml/jackson-core | 964 | Multi-SWE-bench | Apache-2.0 |
| huggingface/transformers | 21969, 24238 | SWE-PolyBench | Apache-2.0 |
| ansible/ansible | two SWE-bench Pro instances | SWE-bench Pro | GPLv3 |
| jqlang/jq | 1793 | Multi-SWE-bench | MIT |
| matplotlib/matplotlib | 25311 | SWE-bench Verified | Matplotlib license (PSF-based), plus the bundled component licenses in its `LICENSE/` folder |
| mui/material-ui | 28813 | SWE-PolyBench | MIT |
| prettier/prettier | 8046 | SWE-PolyBench | MIT |
| psf/requests | 1142 | SWE-bench Verified | Apache-2.0 |
| pydata/xarray | 6721 | SWE-bench Verified | Apache-2.0 |
| pytest-dev/pytest | 7205 | SWE-bench Verified | MIT |
| sphinx-doc/sphinx | 9602, 10614 | SWE-bench Verified | BSD-2-Clause |
| sympy/sympy | 12419, 22714 | SWE-bench Verified | BSD-3-Clause |
| zeromicro/go-zero | 1969 | Multi-SWE-bench | MIT |

Conditions that apply to the redistributed checkouts:

- MIT, BSD and the Matplotlib license: keep the copyright notice and license text. They are kept in each checkout.
- Apache-2.0: keep the license and any NOTICE file, and state changes. They are kept; changes are as above.
- GPLv3 (ansible) and GPLv2 (the zstd option): the planted versions are modified copies. They are distributed in
  source form, under the same license, with this notice of modification and the inserted text recorded in the task's
  labels.

## LegalBench-RAG

- Source: https://github.com/zeroentropy-ai/legalbenchrag at `431bc8f2`; data from `LegalBench-RAG.zip` (sha256
  `27431be37db9b1db23f8ab790a42d076adb1f72d7f9e7562e36a10573405f88d`), the download linked from that README.
- Paper: Pipitone and Houir Alami, LegalBench-RAG, arXiv:2408.10343.
- License: the code is MIT (Copyright (c) 2025 ZeroEntropy), text in `LICENSES/MIT-LegalBench-RAG.txt`. The data
  download has no license file. The documents come from the corpora below, and
  their licenses apply.
- We redistribute the CUAD and MAUD documents (the whole corpus of each, as task workdirs) and the queries and gold
  spans of the 25 drawn LegalBench-RAG queries.
  - CUAD (Hendrycks et al., arXiv:2103.06268): CC BY 4.0 (`LICENSES/CC-BY-4.0.txt`).
  - MAUD (Wang et al., arXiv:2301.00876): CC BY 4.0 (`LICENSES/CC-BY-4.0.txt`).
- LegalBench-RAG also draws on ContractNLI (Koreeda and Manning, arXiv:2110.01799; CC BY 4.0) and PrivacyQA (Ravichander
  et al., arXiv:1911.00841; MIT). No ContractNLI or PrivacyQA document is in our task data.
- Changes: documents renamed to safe file names (mapping in each task's labels); in planted versions, one paragraph
  inserted in one document.

## DocFinQA

- Source: Hugging Face dataset `kensho/DocFinQA`, revision `64ebaff62f692495bcc182f45cf9a9606251b19b`, `test.json`.
- Paper: Reddy et al., DocFinQA, arXiv:2401.06915.
- License: MIT, as its dataset card declares, text in `LICENSES/MIT-DocFinQA.txt`. The card gives no license text
  and names no copyright holder, so that file is the standard MIT text with "the DocFinQA authors" as holder. DocFinQA builds on FinQA (Chen et al., arXiv:2109.00122; MIT,
  Copyright (c) 2021 Zhiyu Chen, text in `LICENSES/MIT-FinQA.txt`).
- The filings are SEC EDGAR filings. The SEC treats information on sec.gov as public information.
- Changes: tabs replaced by spaces as described in `experiments/bench/README.md`; in planted versions, one paragraph
  inserted in the filing.

## AgentDojo

- Source: https://github.com/ethz-spylab/agentdojo at `089ed468cf3ed0322acc66b0211f26d9d90dbf60` (benchmark v1.2.2,
  package 0.1.35).
- Paper: Debenedetti et al., AgentDojo, arXiv:2406.13352.
- License: MIT (Copyright (c) 2024 Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc
  Fischer, and Florian Tramèr), text in `LICENSES/MIT-AgentDojo.txt`.
- Copied text: the attack templates in `experiments/bench/attacks.py` (`important_instructions`,
  `tool_knowledge`, `ignore_previous`, `system_message`, `injecagent`), from `src/agentdojo/attacks/`, with
  file and line provenance in the code. The user name "Emma Johnson" and the injection goals in the attack-wordings
  check are AgentDojo's.
- The AgentDojo run records (`data/agentdojo-gemma4-31b.tar.zst` and `data/agentdojo-deepseek-v4.1-flash.tar.zst`)
  contain AgentDojo's suite data as the agent saw it in tool output. AgentDojo itself is not redistributed;
  `experiments/agentdojo/README.md` says how to clone it.

## Models and services

- Jev is TypeSafe's hosted model (https://docs.typesafe.ai). It is not redistributed. The run records contain its
  scores and request ids.
- The agents were Gemma 4 31B (`google/gemma-4-31b-it`) through OpenRouter, served by Novita; DeepSeek V4.1 Flash
  (`deepseek/deepseek-v4.1-flash`) through OpenRouter, served by Morph; and Mistral Small 3.2 24B
  (`mistralai/Mistral-Small-3.2-24B-Instruct-2506`) on our own vLLM server. No model is redistributed. The run records
  contain their outputs. Use of each model is subject to its own terms.

## Python dependencies

Installed from PyPI by `uv sync` under their own licenses, and not redistributed here. See `pyproject.toml` and
`uv.lock`.
