# Third-party license texts

The license texts for the third-party content this repository redistributes or copies. `THIRD_PARTY_NOTICES.md`
says what each covers. The code and our own data have their own licenses, given there too.

| File | Covers | Where the text came from |
|---|---|---|
| `Apache-2.0-ContextBench.txt` | ContextBench | the `LICENSE` file of github.com/EuniAI/ContextBench at `1436c28a`, unchanged |
| `MIT-AgentDojo.txt` | AgentDojo (the copied attack templates) | the `LICENSE` file of github.com/ethz-spylab/agentdojo at `089ed468`, unchanged |
| `MIT-LegalBench-RAG.txt` | LegalBench-RAG code | the `LICENSE` file of github.com/zeroentropy-ai/legalbenchrag at `431bc8f2`, unchanged |
| `MIT-FinQA.txt` | FinQA, which DocFinQA builds on | the `LICENSE` file of github.com/czyssrs/FinQA (main branch), unchanged |
| `MIT-DocFinQA.txt` | DocFinQA | the standard MIT text, with "Copyright (c) the DocFinQA authors". See the note below |
| `CC-BY-4.0.txt` | CUAD and MAUD documents, and our own data | creativecommons.org/licenses/by/4.0/legalcode.txt, unchanged |

The repository checkouts inside the ContextBench task data keep their own license files in each task's `repo/`.

DocFinQA's dataset card (`kensho/DocFinQA` at `64ebaff6`) gives `license: mit` and nothing more: no license text
and no copyright holder. `MIT-DocFinQA.txt` is therefore the standard MIT text with a generic holder line, "the
DocFinQA authors". It names no holder the card does not name.
