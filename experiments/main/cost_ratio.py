"""jev input tokens spent per agent prompt token saved, per agent model, filter and budget, from the run records.

    uv run --with numpy python experiments/main/cost_ratio.py [--out experiments/main/cost_ratio.csv]

reads the counted records through the analysis loader (analysis/data.py: run_set, load_set, rows_of), so the runs
counted are the paper's: the last record per job that is neither rerun nor stopped, runs that failed on
every try left out. offline, read only on the run folders. needs numpy, which the harness itself does not.

per filter c and budget, over the clean task versions paired with plain on (test bed, item, version), the same pairs
as the token cut (75; fewer where a run failed):
    saving = mean(plain agent prompt tokens - c's agent prompt tokens)   agent prompt = agent calls + compaction calls
    R      = mean(c's jev input tokens) / saving
    break-even agent input price = R x jev's $0.042 per 1M: above it, the agent input tokens saved cost more than jev's
paired bootstrap 95% intervals: 10,000 resamples of task pairs, seed 42. R's interval is the percentiles of the ratio
of resampled means; a resample with no saving has no finite ratio and sits at the top. when the saving's interval
includes 0 there is no ratio. also the paired difference in agent completion (output) tokens, c minus plain.
the same numbers over all versions (clean and planted) are kept as *_all columns.
seed spread: the interval ends under seeds 0-19 are kept as *_s20 columns ("min..max").
a 16K set whose manifest is not in the repo is skipped.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RP = HERE / 'analysis'
sys.path.insert(0, str(RP))
import data as P  # noqa: E402  the analysis loader; importing it runs nothing

JEV_IN = 0.042  # $ per 1M input tokens, jev 1.13; output free
SEED, NB = 42, 10_000
MODELS = {'gemma': 'gemma4-31b.json', 'mistral': 'mistral-small-3.2-24b.json', 'deepseek': 'deepseek-v4.1-flash.json'}
FILTERS = ['relevance', 'relevance-injection']


def boot(x, y, rng):
    """resampled means of x and y over the same pair indices"""
    n = len(x)
    idx = rng.integers(0, n, size=(NB, n))
    return x[idx].mean(axis=1), y[idx].mean(axis=1)


def stats(sav, jev, dout, seed):
    """bootstrap interval ends for the saving, R (ratio of resampled means) and the output-token difference"""
    rng = np.random.default_rng(seed)
    bs, bj = boot(sav, jev, rng)
    s_lo, s_hi = np.percentile(bs, [2.5, 97.5])
    # a resample with no saving has no finite ratio: it sits at the top
    rb = np.where(bs > 0, bj / np.where(bs > 0, bs, 1), np.inf)
    r_lo, r_hi = np.quantile(rb, [0.025, 0.975], method='inverted_cdf') if s_lo > 0 else (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bo, _ = boot(dout, dout, rng)
    o_lo, o_hi = np.percentile(bo, [2.5, 97.5])
    return dict(s_lo=s_lo, s_hi=s_hi, R=jev.mean() / sav.mean(), R_lo=r_lo, R_hi=r_hi, o_lo=o_lo, o_hi=o_hi)


def compute():
    """one row per agent model, budget and filter"""
    rows = []
    for m, cfgf in MODELS.items():
        cfg = json.load(open(RP / 'configs' / cfgf))
        mA, mB, tasks = P.run_set(cfg, ROOT)
        for b, man in (('32K', mA), ('16K', mB)):
            if man is None:
                continue
            S, _, _ = P.load_set(man, ROOT)
            R = P.rows_of(S, man['compaction_budget'], tasks)
            for c in FILTERS:
                row = dict(model=m, budget=b, config=c)
                for kind, suf in (('clean', ''), ('both', '_all')):
                    ctl = P.by_item(R, 'plain', 'all', kind)
                    ctl_runs = P.pick(R, 'plain', 'all', kind)
                    A = P.by_item(R, c, 'all', kind)
                    ks = sorted(set(A) & set(ctl))
                    sav = np.array([ctl[k]['total_prompt_tokens'] - A[k]['total_prompt_tokens'] for k in ks], float)
                    jev = np.array([A[k]['jev_input_tokens'] for k in ks], float)
                    dout = np.array([A[k]['total_completion_tokens'] - ctl[k]['total_completion_tokens'] for k in ks],
                                    float)
                    st = stats(sav, jev, dout, SEED)
                    pos = st['s_lo'] > 0
                    row.update({f'pairs{suf}': len(ks),
                                f'ctl_prompt_run{suf}': round(sum(r['total_prompt_tokens'] for r in ctl_runs) / len(ctl_runs), 1),
                                f'ctl_runs{suf}': len(ctl_runs),
                                f'ctl_prompt_paired{suf}': round(np.mean([ctl[k]['total_prompt_tokens'] for k in ks]), 1),
                                f'cfg_prompt_paired{suf}': round(np.mean([A[k]['total_prompt_tokens'] for k in ks]), 1),
                                f'jev_paired{suf}': round(jev.mean(), 1),
                                f'saving{suf}': round(sav.mean(), 1), f'saving_lo{suf}': round(st['s_lo'], 1),
                                f'saving_hi{suf}': round(st['s_hi'], 1), f'saving_detectable{suf}': bool(pos)})
                    for k in ('R', 'R_lo', 'R_hi'):
                        row[k + suf] = round(st[k], 3) if pos else ''
                    for k, kk in (('breakeven', 'R'), ('breakeven_lo', 'R_lo'), ('breakeven_hi', 'R_hi')):
                        row[k + suf] = round(st[kk] * JEV_IN, 4) if pos else ''
                    row.update({f'dout{suf}': round(dout.mean(), 1), f'dout_lo{suf}': round(st['o_lo'], 1),
                                f'dout_hi{suf}': round(st['o_hi'], 1)})
                    if kind == 'clean':
                        # seed spread of the interval ends, clean only
                        ss = [stats(sav, jev, dout, sd) for sd in range(20)]
                        rng_ = lambda v, f: f'{min(v):{f}}..{max(v):{f}}'
                        row['saving_lo_s20'] = rng_([x['s_lo'] for x in ss], '.0f')
                        row['saving_hi_s20'] = rng_([x['s_hi'] for x in ss], '.0f')
                        row['detectable_s20'] = f"{sum(x['s_lo'] > 0 for x in ss)}/20"
                        if pos:
                            row['R_lo_s20'] = rng_([x['R_lo'] for x in ss], '.3f')
                            row['R_hi_s20'] = rng_([x['R_hi'] for x in ss], '.3f')
                            row['breakeven_lo_s20'] = rng_([x['R_lo'] * JEV_IN for x in ss], '.4f')
                            row['breakeven_hi_s20'] = rng_([x['R_hi'] * JEV_IN for x in ss], '.4f')
                        else:
                            row.update(R_lo_s20='', R_hi_s20='', breakeven_lo_s20='', breakeven_hi_s20='')
                row.update(seed=SEED, resamples=NB, source=(
                    f'records via analysis/data.py (configs/{cfgf}), {b}: total_prompt_tokens, jev_input_tokens, '
                    f'total_completion_tokens per counted run, paired on (bench, item, version), clean versions (no '
                    f'suffix) and all versions (_all)'))
                rows.append(row)
    return rows


def write(rows, out):
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model']:8} {r['budget']} {r['config']:19} clean n={r['pairs']} saving {r['saving']:,.0f} "
              f"({r['saving_lo']:,.0f} to {r['saving_hi']:,.0f}) [{r['detectable_s20']}]  R {r['R']} ({r['R_lo']} to "
              f"{r['R_hi']})  break-even ${r['breakeven']} ({r['breakeven_lo']} to {r['breakeven_hi']})  "
              f"dout {r['dout']:,.0f} ({r['dout_lo']:,.0f} to {r['dout_hi']:,.0f})")
    print(f'{out}: {len(rows)} rows')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=HERE / 'cost_ratio.csv')
    write(compute(), ap.parse_args().out)


if __name__ == '__main__':
    main()
