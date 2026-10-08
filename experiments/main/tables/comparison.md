# AgentDojo: our runs next to published defenses

This is the paper's Table 3. Values are percentages, no defense -> with defense. Benign utility is without attack,
attacked utility under attack. n/r: not reported. [dagger] Computed from reported counts, 44 and 0 of 949.

| Defense | Agent | Benign utility | Attacked utility | Attack success |
|---|---|---|---|---|
| This work | Gemma 4 31B | 94.8 -> 92.8 | 80.5 -> 91.6 | 27.3 -> 0.1 |
| This work | DeepSeek V4.1 Flash | 91.8 -> 95.9 | 92.4 -> 91.5 | 1.7 -> 0.0 |
| AgentSentry [zhang2026agentsentry] | GPT-4o | 78.35 -> 78.35 | 36.84 -> 75.49 | 33.23 -> 0.00 |
| AgentSentry [zhang2026agentsentry] | Qwen3-Max | 85.57 -> 83.51 | 56.24 -> 86.92 | 33.38 -> 0.00 |
| DataFilter [wang2026datafilter] | GPT-4o | 81.4 -> 79.4 | 46.7 -> 72.5 | 42.2 -> 0.2 |
| PromptArmor [shi2025promptarmor] | GPT-4.1 | n/r | 64.27 -> 72.02 | 54.53 -> 0.00 |
| &nbsp;&nbsp;repr. in [wang2026datafilter] | GPT-4o | 81.4 -> 72.2 | 46.7 -> 67.1 | 42.2 -> 2.5 |
| CommandSans* [das2025commandsans] | GPT-4o | 69.07 -> 77.32 | 46.89 -> 63.75 | 34.67 -> 3.48 |
| CaMeL [debenedetti2025camel] | Claude 3.5 Sonnet | 90.72 -> 63.92 | 86.83 -> 66.70 | 4.6 [dagger] -> 0 |

The published rows are each paper's reported numbers, not rerun. Agent models, temperatures, benchmark versions and
attacks differ, so the table is not a ranking. PromptArmor's own row pools four attacks over 629 cases. Its indented
row is PromptArmor with a GPT-4.1 detector and a modified prompt, reimplemented by the DataFilter authors.
CommandSans* is that paper's variant.

Our two rows come from `experiments/agentdojo/` (see its RESULTS.md); `experiments/main/paper_tables.py` writes them
to `t4_agentdojo.csv`. Citation keys are in `references.bib`:

- zhang2026agentsentry: AgentSentry, https://arxiv.org/abs/2602.22724
- wang2026datafilter: DataFilter, https://arxiv.org/abs/2510.19207
- shi2025promptarmor: PromptArmor, https://arxiv.org/abs/2507.15219
- das2025commandsans: CommandSans, https://arxiv.org/abs/2510.08829
- debenedetti2025camel: CaMeL, https://arxiv.org/abs/2503.18813
- debenedetti2024agentdojo: AgentDojo, https://arxiv.org/abs/2406.13352
