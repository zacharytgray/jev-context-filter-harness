"""cross-model look at why relevance cuts tokens for one agent model and not another: how much raw tool output the
plain loop pulls in, how it reads, and the paired token cut. compares the three agent models' 32K and 16K runs:
gemma 4 31b, mistral small 3.2 24b, deepseek v4.1 flash.

    uv run python experiments/main/cross_model/xmodel.py [--out DIR]     # or plain python3, stdlib only

offline, read only on the run folders. run selection is the analysis loader's own (experiments/main/analysis/data.py,
imported): per job the last record that is neither rerun nor stopped, and runs that still failed on the rerun rule's
last try are left out. writes out.json and per_task.csv to --out, default next to this file.
"""
import argparse
import csv
import json
import math
import os
import random
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
sys.path.insert(0, str(WT / 'experiments' / 'main' / 'analysis'))
import data as D  # noqa: E402

CONFIGS = [('gemma', 'Gemma 4 31B', 'gemma4-31b.json'),
           ('mistral', 'Mistral Small 3.2 24B', 'mistral-small-3.2-24b.json'),
           ('deepseek', 'DeepSeek V4.1 Flash', 'deepseek-v4.1-flash.json')]
FILTERS = ['relevance', 'injection', 'relevance-injection']
PASS_THROUGH = 4000  # s1harness/pipeline.py PASS_THROUGH_CHARS: relevance only chunks outputs longer than this
SEED, NBOOT, NPERM = 20261004, 10_000, 10_000
FIELD = 'total_prompt_tokens'  # agent llm prompt tokens: every agent and compaction call, no jev

# bash: classify the first real command of the call (after cd/export/echo segments) by its first word
SEARCH = {'grep', 'egrep', 'fgrep', 'rg', 'ag', 'ack'}
SLICE = {'sed', 'head', 'tail', 'awk', 'cut', 'nl'}
DUMP = {'cat', 'less', 'more', 'bat', 'tac'}
SKIP = re.compile(r'^(?:#|(?:cd|pwd|export|echo|set|source|\.|true)\b)')  # comment lines are skipped too
HEREDOC_WRITE = re.compile(r'^cat\s*(?:>|<<)')  # cat <<EOF > file, cat > file <<EOF: a write, not a read


def first_segment(cmd):
    segs = [s for s in re.split(r'\s*(?:&&|;|\|\||\n)\s*', cmd.strip()) if s and not SKIP.match(s)]
    return segs[0] if segs else None


def first_words(cmd):
    seg = first_segment(cmd)
    if seg is None:
        return None, []
    segs = [seg]
    pipe = [p.strip().split() for p in segs[0].split('|')]
    words = [os.path.basename(p[0]) if p else '' for p in pipe]
    if words[0] == 'timeout' and len(pipe[0]) > 2:
        words[0] = os.path.basename(pipe[0][2])
    if words[0] == 'git' and len(pipe[0]) > 1 and pipe[0][1] == 'grep':
        words[0] = 'grep'
    return words[0], words[1:]


def classify_bash(cmd):
    """search (grep, rg), slice (sed, head, tail, awk), dump (cat) or other. a cat piped into grep is a search,
    into head or sed a slice. a heredoc write with cat is other"""
    w, rest = first_words(cmd)
    if w is None:
        return 'other', ''
    if w == 'cat' and HEREDOC_WRITE.match(first_segment(cmd)):
        return 'other', w
    if w in DUMP:
        if any(x in SEARCH for x in rest):
            return 'search', w
        if any(x in SLICE for x in rest):
            return 'slice', w
        return 'dump', w
    if w in SEARCH:
        return 'search', w
    if w in SLICE:
        return 'slice', w
    return 'other', w


def classify_read(args, out):
    """whole: no range given (the harness's own test, loop.py: start_line or end_line). ranged otherwise. a ranged read
    from line 1 whose range ran past the file's end got the whole file anyway: whole_by_range, kept apart in the
    counts and counted as ranged in the shares and the raw chars"""
    sl, el = args.get('start_line'), args.get('end_line')
    if not (sl or el):
        return 'whole'
    try:
        if (sl or 1) <= 1 and el and out and out.get('ok') and out.get('raw_lines') is not None \
                and int(out['raw_lines']) < int(el):
            return 'whole_by_range'
    except (TypeError, ValueError):
        pass
    return 'ranged'


def trace_calls(run):
    """read and bash calls of one run's trace, in order, each with its output event. raw_chars is the tool's own
    output before any filter or the 50,000-character cap: for read the numbered lines the read tool returned, for
    bash stdout plus stderr, stripped, without the exit code line. a call that never ran (bad arguments) has no
    output event and no raw chars"""
    calls, outs = [], {}
    for d in D.jl(Path(run) / 'trace.jsonl'):
        ev = d.get('event')
        if ev == 'calls':
            calls += [c for c in d['calls'] if c.get('tool') in ('read', 'bash')]
        elif ev == 'output':
            outs[d['call_id']] = d
    res = []
    for c in calls:
        o = outs.get(c['call_id'])
        a = c.get('args') or {}
        if c['tool'] == 'read':
            kind, word = classify_read(a, o), None
        else:
            kind, word = classify_bash(str(a.get('command') or ''))
        res.append(dict(tool=c['tool'], kind=kind, word=word, ran=o is not None, ok=bool(o and o.get('ok')),
                        raw=(o or {}).get('raw_chars'), pass_through=(o or {}).get('pass_through'),
                        entered=(o or {}).get('entered_chars')))
    return res


def run_stats(calls):
    rd = [c for c in calls if c['tool'] == 'read']
    bs = [c for c in calls if c['tool'] == 'bash']
    k = Counter(c['kind'] for c in calls)
    rk = lambda *kinds: sum(c['raw'] or 0 for c in calls if c['kind'] in kinds)
    return dict(raw_chars=sum(c['raw'] or 0 for c in calls), raw_read=sum(c['raw'] or 0 for c in rd),
                raw_bash=sum(c['raw'] or 0 for c in bs), reads=len(rd), bash=len(bs),
                # raw chars by class, for the paper's reading figure
                raw_read_whole=rk('whole'), raw_read_ranged=rk('ranged', 'whole_by_range'),
                raw_bash_snippet=rk('search', 'slice', 'dump'), raw_bash_other=rk('other'),
                read_whole=k['whole'], read_whole_by_range=k['whole_by_range'], read_ranged=k['ranged'],
                bash_search=k['search'], bash_slice=k['slice'], bash_dump=k['dump'],
                bash_snippet=k['search'] + k['slice'] + k['dump'], bash_other=k['other'],
                outputs=sum(c['ran'] for c in calls), outputs_gt4k=sum(c['ran'] and (c['raw'] or 0) > PASS_THROUGH for c in calls))


def summ(xs):
    xs = [x for x in xs if x is not None]
    return dict(n=len(xs), mean=st.mean(xs) if xs else None, median=st.median(xs) if xs else None)


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for q in range(i, j + 1):
            r[o[q]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    n = len(x); mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else float('nan')


def spearman(x, y, seed=SEED):
    """rho with a two-sided permutation p (fixed seed)"""
    if len(x) < 3:
        return dict(n=len(x), rho=None, p_perm=None)
    rx, ry = ranks(x), ranks(y)
    rho = pearson(rx, ry)
    rng = random.Random(seed); ry2 = list(ry); hits = 0
    for _ in range(NPERM):
        rng.shuffle(ry2)
        hits += abs(pearson(rx, ry2)) >= abs(rho) - 1e-12
    return dict(n=len(x), rho=rho, p_perm=(hits + 1) / (NPERM + 1))


def ratio_cut(pairs, seed=SEED):
    """pairs of (filter tokens, plain tokens) on the same clean task. cut = 1 - mean(filter)/mean(plain). t interval:
    the paired t interval on the difference, over mean(plain) (ignores the denominator's own uncertainty).
    bootstrap: tasks resampled with their pairs, percentile 95% interval of the cut"""
    n = len(pairs)
    a = [p[0] for p in pairs]; b = [p[1] for p in pairs]
    ma, mb = st.mean(a), st.mean(b)
    t = D.tmean([x - y for x, y in pairs])
    cut = 1 - ma / mb
    rng = random.Random(seed); boots = []
    for _ in range(NBOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = sum(a[i] for i in idx); sb = sum(b[i] for i in idx)
        boots.append(1 - sa / sb)
    boots.sort()
    return dict(n=n, mean_filter=ma, mean_plain=mb, mean_diff=t['mean'], diff_lo=t['lo'], diff_hi=t['hi'], cut=cut,
                t_lo=-t['hi'] / mb, t_hi=-t['lo'] / mb,
                boot_lo=boots[int(0.025 * NBOOT)], boot_hi=boots[int(0.975 * NBOOT) - 1])


def load(cfgname):
    cfg = json.load(open(WT / 'experiments' / 'main' / 'analysis' / 'configs' / cfgname))
    mA, mB, tasks = D.run_set(cfg, WT)
    SA, _, FA = D.load_set(mA, WT)
    RA = D.rows_of(SA, mA['compaction_budget'], tasks)
    RB = []
    if mB:
        SB, _, _ = D.load_set(mB, WT)
        RB = D.rows_of(SB, mB['compaction_budget'], tasks)
    return cfg, RA, RB, len(FA)


def paired_rows(R, c):
    A, P = D.by_item(R, c, 'all', 'clean'), D.by_item(R, 'plain', 'all', 'clean')
    return [(A[k], P[k]) for k in sorted(set(A) & set(P)) if A[k].get(FIELD) is not None and P[k].get(FIELD)]


def main(out_dir=HERE):
    out, csv_rows, pooled_x, pooled_y = {}, [], [], []
    for key, name, cfgname in CONFIGS:
        cfg, RA, RB, nfail = load(cfgname)
        M = out[key] = dict(model=name, config=cfgname, tag=cfg['runs']['tag'], counted_32k=len(RA),
                            failed_every_try_32k=nfail, counted_16k=len(RB))
        # 1. per plain run: raw chars and how it read
        per = {}
        for r in D.pick(RA, 'plain'):
            cs = trace_calls(r['_run'])
            per[r['t']] = dict(run_stats(cs), kind=r['kind'], b=r['b'], item=r['item'])
        fields = [k for k in next(iter(per.values())) if k not in ('kind', 'b', 'item')]
        M['plain_per_run'] = {v: dict(runs=sum(1 for p in per.values() if v == 'all' or p['kind'] == v),
                                      **{f: summ([p[f] for p in per.values() if v == 'all' or p['kind'] == v]) for f in fields})
                              for v in ('clean', 'all')}
        for v in ('clean', 'all'):
            ps = [p for p in per.values() if v == 'all' or p['kind'] == v]
            nr, nb = sum(p['reads'] for p in ps), sum(p['bash'] for p in ps)
            M['plain_per_run'][v]['shares'] = dict(
                read_whole=sum(p['read_whole'] for p in ps) / nr if nr else None,
                read_ranged=sum(p['read_ranged'] + p['read_whole_by_range'] for p in ps) / nr if nr else None,
                bash_snippet=sum(p['bash_snippet'] for p in ps) / nb if nb else None,
                read_share_of_raw=sum(p['raw_read'] for p in ps) / max(1, sum(p['raw_chars'] for p in ps)))
        # 2. per clean task: plain raw chars against the relevance token difference
        pr = paired_rows(RA, 'relevance')
        xs, ys, cuts = [], [], []
        for a, p in pr:
            raw = per[p['t']]['raw_chars']
            d = a[FIELD] - p[FIELD]
            csv_rows.append(dict(model=key, task=p['t'], testbed=p['b'], plain_raw_chars=raw, plain_tokens=p[FIELD],
                                 rel_tokens=a[FIELD], diff=d, cut_pct=round(100 * (1 - a[FIELD] / p[FIELD]), 2)))
            xs.append(raw); ys.append(d); cuts.append(1 - a[FIELD] / p[FIELD])
        pooled_x += xs; pooled_y += ys
        M['rel_clean'] = dict(n=len(pr), plain_tok=summ([p[FIELD] for _, p in pr]), rel_tok=summ([a[FIELD] for a, _ in pr]),
                              diff=D.tmean(ys), cut_mean=st.mean(cuts), cut_median=st.median(cuts),
                              plain_raw_chars=summ(xs))
        # 3. spearman within model, and the share of relevance outputs long enough to be filtered
        M['spearman_raw_vs_diff'] = spearman(xs, ys)
        for arm in ('relevance', 'plain'):
            for v in ('clean', 'all'):
                cs = [c for r in D.pick(RA, arm, 'all', None if v == 'all' else v) for c in trace_calls(r['_run']) if c['ran']]
                ok = [c for c in cs if c['ok']]
                big = [c for c in ok if not c['pass_through']]
                M.setdefault('gt4k', {}).setdefault(arm, {})[v] = dict(
                    outputs=len(cs), ok_outputs=len(ok), filterable=len(big), share=len(big) / len(cs) if cs else None,
                    char_share=sum(c['raw'] for c in big) / max(1, sum(c['raw'] or 0 for c in cs)),
                    kept_chars=(sum(c['entered'] or 0 for c in big) / max(1, sum(c['raw'] for c in big))) if arm == 'relevance' else None)
        # 4. ratio-of-means cut, each filter against plain, 32K and 16K
        M['ratio_cut'] = {'32K': {c: ratio_cut([(a[FIELD], p[FIELD]) for a, p in paired_rows(RA, c)]) for c in FILTERS}}
        if RB:
            M['ratio_cut']['16K'] = {c: ratio_cut([(a[FIELD], p[FIELD]) for a, p in paired_rows(RB, c)]) for c in FILTERS}
    out['pooled'] = dict(spearman_raw_vs_diff=spearman(pooled_x, pooled_y))
    out['method'] = dict(field=FIELD, seed=SEED, n_boot=NBOOT, n_perm=NPERM, pass_through=PASS_THROUGH)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_dir / 'out.json', 'w'), indent=1)
    with open(out_dir / 'per_task.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'task', 'testbed', 'plain_raw_chars', 'plain_tokens', 'rel_tokens', 'diff', 'cut_pct'])
        w.writeheader(); w.writerows(csv_rows)
    report(out)


def report(o):
    f0 = lambda x: '-' if x is None else f'{x:,.0f}'
    f1 = lambda x: '-' if x is None else f'{x:.1f}'
    pc = lambda x: '-' if x is None else f'{100 * x:.1f}%'
    for v in ('clean', 'all'):
        print(f'\n1. plain runs, {v} versions: mean / median per run')
        print('model     runs  raw chars              reads        whole        ranged       bash         snippet      other')
        for k in ('gemma', 'mistral', 'deepseek'):
            P = o[k]['plain_per_run'][v]
            g = lambda f: f"{f1(P[f]['mean'])}/{f1(P[f]['median'])}"
            print(f"{k:9s} {P['runs']:4d}  {f0(P['raw_chars']['mean'])}/{f0(P['raw_chars']['median']):10s}"
                  f"  {g('reads'):11s}  {g('read_whole'):11s}  {g('read_ranged'):11s}  {g('bash'):11s}  {g('bash_snippet'):11s}  {g('bash_other')}"
                  f"   (whole by range {g('read_whole_by_range')}; search/slice/dump {g('bash_search')} {g('bash_slice')} {g('bash_dump')})")
    print('\n2-3. relevance vs plain, clean pairs')
    for k in ('gemma', 'mistral', 'deepseek'):
        R = o[k]['rel_clean']; s = o[k]['spearman_raw_vs_diff']; g = o[k]['gt4k']
        print(f"{k:9s} n={R['n']} plain mean {f0(R['plain_tok']['mean'])} med {f0(R['plain_tok']['median'])}  rel mean {f0(R['rel_tok']['mean'])} med {f0(R['rel_tok']['median'])}"
              f"  diff {f0(R['diff']['mean'])} ({f0(R['diff']['lo'])} to {f0(R['diff']['hi'])})  cut mean {pc(R['cut_mean'])} med {pc(R['cut_median'])}"
              f"  rho(raw,diff) {s['rho']:+.3f} p {s['p_perm']:.4f}"
              f"  rel outputs >4k: clean {pc(g['relevance']['clean']['share'])} ({g['relevance']['clean']['filterable']}/{g['relevance']['clean']['outputs']}), all {pc(g['relevance']['all']['share'])}"
              f"; plain >4k clean {pc(g['plain']['clean']['share'])}")
    p = o['pooled']
    print(f"pooled    rho(raw,diff) {p['spearman_raw_vs_diff']['rho']:+.3f} p {p['spearman_raw_vs_diff']['p_perm']:.4f} n={p['spearman_raw_vs_diff']['n']}")
    print('\n4. cut = 1 - mean(filter)/mean(plain), clean pairs')
    for k in ('gemma', 'mistral', 'deepseek'):
        for b, rows in o[k]['ratio_cut'].items():
            for c, r in rows.items():
                print(f"{k:9s} {b} {c:20s} n={r['n']} plain {f0(r['mean_plain'])} filt {f0(r['mean_filter'])}  cut {pc(r['cut'])}"
                      f"  t {pc(r['t_lo'])} to {pc(r['t_hi'])}  boot {pc(r['boot_lo'])} to {pc(r['boot_hi'])}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='cross-model look at the relevance token cut')
    ap.add_argument('--out', default=str(HERE), help='folder for out.json and per_task.csv, default this folder')
    main(ap.parse_args().out)
