"""compaction per agent model, compaction threshold and configuration: each filter against Control, and 16K against
32K, from the counted records. the paper's compaction section and the supplement's compaction rows.

    python3 experiments/main/compaction.py [--csv FILE]     # stdlib only

the analysis loader's own selection (analysis/data.py: run_set, load_set, rows_of), the same as table 2's comp_runs
and comps: per job the last record that is neither rerun nor stopped, runs that failed on every try left out. all
task versions, clean and planted, paired by task version (test bed, item, version) as data.py by_item does.
    share compacted: runs with at least one compaction. paired difference with newcombe's method 10 (data.py
        newcombe_paired, wilson limits and the phi adjustment)
    compactions per run: mean, and the paired mean difference with a t 95% interval (data.py tmean)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'analysis'))
import data as P  # noqa: E402

MODELS = {'gemma': 'gemma4-31b.json', 'mistral': 'mistral-small-3.2-24b.json', 'deepseek': 'deepseek-v4.1-flash.json'}
CF = ['plain', 'relevance', 'injection', 'relevance-injection']


def sets(cfgname):
    cfg = json.load(open(HERE / 'analysis' / 'configs' / cfgname))
    mA, mB, tasks = P.run_set(cfg, P.WT)
    out = {}
    for b, m in (('32K', mA), ('16K', mB)):
        S, _, _ = P.load_set(m, P.WT)
        out[b] = P.rows_of(S, m['compaction_budget'], tasks)
    return out


def compare(X, Y):
    """X against Y, both {key: row}, paired on shared keys"""
    ks = sorted(set(X) & set(Y))
    cx = [X[k]['compactions'] > 0 for k in ks]
    cy = [Y[k]['compactions'] > 0 for k in ks]
    both = sum(a and b for a, b in zip(cx, cy)); o1 = sum(a and not b for a, b in zip(cx, cy))
    o2 = sum(b and not a for a, b in zip(cx, cy)); ne = len(ks) - both - o1 - o2
    d, lo, hi = P.newcombe_paired(both, o1, o2, ne)
    md = P.tmean([X[k]['compactions'] - Y[k]['compactions'] for k in ks])
    return dict(n=len(ks), kx=sum(cx), ky=sum(cy), both=both, only_x=o1, only_y=o2, neither=ne,
                share_x=sum(cx) / len(ks), share_y=sum(cy) / len(ks), d=d, lo=lo, hi=hi,
                mean_x=sum(X[k]['compactions'] for k in ks) / len(ks), mean_y=sum(Y[k]['compactions'] for k in ks) / len(ks),
                md=md['mean'], md_lo=md['lo'], md_hi=md['hi'])


def det(lo, hi):
    return 'detectable' if lo > 0 or hi < 0 else 'no detectable difference'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, help='also write every comparison to this csv')
    a = ap.parse_args()
    rows = []
    for mk, cfgname in MODELS.items():
        R = sets(cfgname)
        print(f'## {mk}')
        print('totals (all runs, as table 2): threshold config runs comp_runs comps share mean')
        for b in ('32K', '16K'):
            for c in CF:
                rs = P.pick(R[b], c)
                cr, cs = sum(r['compactions'] > 0 for r in rs), sum(r['compactions'] for r in rs)
                print(f'  {b} {c:20s} {len(rs):4d} {cr:4d} {cs:4d} {cr / len(rs):.3f} {cs / len(rs):.3f}')
        print('filter against Control, paired by task version: share compacted (filter vs Control), difference [95% CI newcombe],'
              ' mean compactions per run (filter vs Control), difference [95% CI t]')
        for b in ('32K', '16K'):
            ctl = P.by_item(R[b], 'plain', 'all', 'both')
            for c in CF[1:]:
                x = compare(P.by_item(R[b], c, 'all', 'both'), ctl)
                rows.append(dict(model=mk, kind='vs_control', threshold=b, config=c, **x))
                print(f"  {b} {c:20s} n={x['n']:3d} {x['kx']:3d}/{x['n']} vs {x['ky']:3d}/{x['n']}  "
                      f"{x['share_x']:.3f} vs {x['share_y']:.3f}  d={x['d']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] {det(x['lo'], x['hi'])}  "
                      f"| mean {x['mean_x']:.3f} vs {x['mean_y']:.3f} d={x['md']:+.3f} [{x['md_lo']:+.3f}, {x['md_hi']:+.3f}] {det(x['md_lo'], x['md_hi'])}")
        print('16K against 32K, same configuration, paired by task version')
        for c in CF:
            x = compare(P.by_item(R['16K'], c, 'all', 'both'), P.by_item(R['32K'], c, 'all', 'both'))
            rows.append(dict(model=mk, kind='16K_vs_32K', threshold='16K-32K', config=c, **x))
            print(f"  {c:20s} n={x['n']:3d} 32K {x['ky']:3d}/{x['n']} -> 16K {x['kx']:3d}/{x['n']}  "
                  f"d={x['d']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] {det(x['lo'], x['hi'])}  "
                  f"| mean {x['mean_y']:.3f} -> {x['mean_x']:.3f} d={x['md']:+.3f} [{x['md_lo']:+.3f}, {x['md_hi']:+.3f}] {det(x['md_lo'], x['md_hi'])}")
        print()
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
