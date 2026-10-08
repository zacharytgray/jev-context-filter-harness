"""the measures of one agent model's runs: its 600 runs at 32K (set A) and its 16K set (set B), from the run records.
the paper's Table 2 and Figure 3 numbers come from here (experiments/main/paper_tables.py runs it for all three).

    python3 experiments/main/analysis/data.py [--config configs/gemma4-31b.json] [--root ROOT] [--b16k-manifest PATH]
        [--out out-<config name>/data.json]

the config (configs/*.json, default gemma4-31b.json) names the model, its two manifests and run tags and the expected
counts. its paths are relative to the repo root unless absolute.

set A is the 32K manifest: its played cells under experiments/runs-<tag>/<suite> and its `reused` runs, if any. set B
is the 16K manifest: its played runs under its own runs folder and its `reused` runs that carry over from set A
(their records and run dirs, repo-relative or absolute). per (configuration, task, rep) the counted record is the
last that is neither rerun nor stopped. a run that failed on every try is left out of every measure and listed in
`final`. a record's run_dir is relative to the repo root its records file lives in. pairs need both runs. offline and
read only on the run folders.

intervals are 95%: a mean is mean +- t(0.975, n - 1) sd / sqrt(n), not clipped; a rate is wilson's; a paired rate is
newcombe's paired score interval (method 10, phi adjusted). agentdojo() gives the AgentDojo rates of one run folder.
"""
import argparse
import glob
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
CONFIG = HERE / 'configs' / 'gemma4-31b.json'
# the repo root this folder sits in
WT = HERE.resolve().parents[2]
CF = ['plain', 'relevance', 'injection', 'relevance-injection']  # configurations shown, in this order
BN = ['contextbench', 'legalbenchrag', 'docfinqa']
SC = BN + ['all']  # test beds, then pooled
PAIRS = [('relevance', 'plain'), ('injection', 'plain'), ('relevance-injection', 'plain')]
VERSIONS = ['clean', 'planted', 'both']
Z = 1.959964
# t(0.975, df) for df 1 to 30; past 30 the cornish-fisher expansion
T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


def tq(df):
    if df <= 30:
        return T975[df - 1]
    z, v = Z, df
    return (z + (z ** 3 + z) / (4 * v) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * v ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * v ** 3)
            + (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / (92160 * v ** 4))


def tmean(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if not n:
        return dict(n=0, mean=None, sd=None, lo=None, hi=None)
    m = sum(xs) / n
    if n < 2:
        return dict(n=1, mean=m, sd=None, lo=None, hi=None)
    sd = st.stdev(xs); h = tq(n - 1) * sd / math.sqrt(n)
    return dict(n=n, mean=m, sd=sd, lo=m - h, hi=m + h)


def wilson(k, n):
    if not n:
        return dict(k=k, n=0, rate=None, lo=None, hi=None)
    p = k / n; d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return dict(k=k, n=n, rate=p, lo=max(0.0, c - h), hi=min(1.0, c + h))


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def mx(xs):
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def jl(p):
    out = []
    try:
        for l in open(p):
            try:
                out.append(json.loads(l))
            except ValueError:
                pass  # a half-written last line
    except FileNotFoundError:
        pass
    return out


def wt_of(p):
    p = Path(p).resolve()
    for q in [p, *p.parents]:
        if q.name == 'experiments':
            return q.parent
    raise SystemExit(f'{p} is not under an experiments folder')


_files = {}


def jobs_in(path):
    """per (configuration, task, rep) of a records.jsonl: every record in order"""
    path = str(path)
    if path not in _files:
        jobs = defaultdict(list)
        for r in jl(path):
            if isinstance(r, dict) and r.get('task') is not None:
                jobs[(r.get('configuration') or r.get('arm'), r['task'], r.get('rep'))].append(r)
        _files[path] = jobs
    return _files[path]


def run_dir_of(d):
    d = Path(d)
    if not (d / 'llm.jsonl').exists() and (d / 'run').is_dir():
        subs = sorted(p for p in (d / 'run').iterdir() if p.is_dir())
        if subs:
            return subs[-1]
    return d


def counted(path, key, run_dir=None):
    """the counted record of a job (the last neither rerun nor stopped), or for a reused run the record with its
    run dir; plus the job's rerun and stopped records before it, and whether one waits after it"""
    rs = jobs_in(path).get(key, [])
    wt = wt_of(path)
    idx = [i for i, r in enumerate(rs) if not r.get('rerun') and not r.get('stopped')]
    if run_dir:
        hit = [i for i in idx if str(wt / rs[i].get('run_dir', '')) == str(Path(run_dir))]
        idx = hit or idx
    if not idx:
        return None, dict(waiting=bool(rs), reruns=sum(bool(r.get('rerun')) for r in rs))
    i = idx[-1]
    r = dict(rs[i])
    r['_run'] = str(run_dir_of(wt / r['run_dir'])) if r.get('run_dir') else None
    return r, dict(waiting=bool(rs[i + 1:]), reruns=sum(bool(x.get('rerun')) for x in rs[:i]),
                   stopped=sum(bool(x.get('stopped')) for x in rs[:i]))


def jobs_of(m):
    return [(arm, task, rep) for c in m['cells'] for rep in range(c.get('start', 0), c.get('start', 0) + c.get('repeats', 1))
            for task in c['tasks'] for arm in c['arms']]


_prompts = {}


def prompts(run):
    """the exact prompt_tokens of a run's agent turn calls"""
    if run not in _prompts:
        p = Path(run or '') / 'llm.jsonl'
        _prompts[run] = [x.get('prompt_tokens') or 0 for x in jl(p) if x.get('kind') == 'turn'] if run and p.exists() else []
    return _prompts[run]


FAIL_ENDS = ('error', 'overflow', 'crash', 'timeout')
final_fail = lambda r: r.get('end') in FAIL_ENDS or bool(r.get('crash'))


def load_set(m, root):
    """the counted record of every job of a manifest, played and reused, keyed (configuration, task, rep); and how
    far the set has got"""
    runs = root / 'experiments' / f"runs-{m['tag']}" / m['suite']
    out, prog, final = {}, {c: Counter() for c in CF}, {}
    # earlier failed plays per job, moved aside under the rerun rule
    tried = Counter((c, r.get('task'), r.get('rep')) for c in CF for r in jl(runs / c / 'records.failed.jsonl') if isinstance(r, dict))
    for key in jobs_of(m):
        r, why = counted(runs / key[0] / 'records.jsonl', key)
        prog[key[0]]['played_total'] += 1
        prog[key[0]]['reruns'] += why.get('reruns', 0)
        if r is None:
            prog[key[0]]['waiting'] += why['waiting']
            continue
        r['_reused'] = False; r['_reruns'] = why['reruns']
        prog[key[0]]['played_done'] += 1
        if final_fail(r):  # still failed on the rerun rule's last try: reported, not data
            r['_tries'] = 1 + tried[key]
            final[key] = r
            continue
        out[key] = r
        prog[key[0]]['crash'] += bool(r.get('crash'))
        prog[key[0]]['timeout'] += 'timeout' in str(r.get('crash') or '')
    for x in m.get('reused') or []:
        key = (x['configuration'], x['task'], x['rep'])
        r, why = counted(resolve(root, x['records']), key, x.get('run_dir') and resolve(root, x['run_dir']))
        prog[key[0]]['reused_total'] += 1
        if r is None:
            continue
        r['_reused'] = True; r['_reruns'] = 0
        prog[key[0]]['reused_done'] += 1
        if final_fail(r):  # a 32K run that failed on every try, carried into the 16K set as a failure
            r['_tries'] = (x.get('failed') or {}).get('tries', 1)
            final[key] = r
            continue
        out[key] = r
    live = 0
    for d in glob.glob(str(runs / '*' / '*' / 'r*' / 'run' / '*')):
        if Path(d, 'llm.jsonl').exists() and not Path(d, 'result.json').exists():
            live += 1
    return out, dict(per={c: dict(prog[c]) for c in CF}, live=live), final


num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)


def rows_of(S, budget, tasks):
    out = []
    for (c, t, rep), r in S.items():
        tk = tasks.get(t, {})
        p = prompts(r['_run'])
        out.append(dict(r, c=c, t=t, rep=rep, b=r.get('bench') or tk.get('benchmark'), item=tk.get('item', t.rsplit('-', 1)[0]),
                        kind='clean' if (tk.get('kind') or r.get('bench_kind')) == 'clean' else 'planted',
                        score=r.get('score') if num(r.get('score')) else None, reached=bool(r.get('planted_reached')),
                        over=sum(x > budget for x in p), over_before_final=sum(x > budget for x in p[:-1])))
    return out


def pick(R, c=None, s='all', kind=None):
    return [x for x in R if (c is None or x['c'] == c) and (s == 'all' or x['b'] == s) and (kind in (None, 'both') or x['kind'] == kind)]


def by_item(R, c, s, kind):
    return {(x['b'], x['item'], x['kind']): x for x in pick(R, c, s, kind)}


def paired(a, b, f):
    ds, dropped = [], 0
    for k in sorted(set(a) & set(b)):
        d = f(a[k], b[k])
        if d is None:
            dropped += 1
        else:
            ds.append(d)
    return ds, dropped


def diff(field):
    return lambda x, y: None if x[field] is None or y[field] is None else float(x[field]) - float(y[field])


def red(x, y):
    t, p = x.get('total_prompt_tokens'), y.get('total_prompt_tokens')
    return None if t is None or not p else 1 - t / p


updown = lambda ds: dict(up=sum(d > 0 for d in ds), same=sum(d == 0 for d in ds), down=sum(d < 0 for d in ds))


def compute(R, F=()):
    """the measures over the counted rows R. F are the runs that still failed on the rerun rule's last try: no measure
    uses them; they are listed (final) and counted per configuration (crashes)"""
    D = dict(final=[dict(c=r['c'], t=r['t'], b=r['b'], kind=r['kind'], end=r.get('end'), error=r.get('error') or r.get('crash'),
                         tries=r.get('_tries', 1), jev=r.get('jev_input_tokens') or 0) for r in F],
             runs=len(R), score=[], tokens=[], perrun=[], config=[], planted=[], paired=[], ptok=[], rates=[], util=[])
    for s in SC:
        for c in CF:
            rs = pick(R, c, s, 'clean'); xs = [r['score'] for r in rs if r['score'] is not None]
            D['score'].append(dict(b=s, c=c, **tmean(xs), runs=len(rs), unscored=len(rs) - len(xs),
                                   zeros=sum(x == 0 for x in xs), top=sum(x >= 0.9 for x in xs)))
            pt = [r['total_prompt_tokens'] for r in rs]
            D['tokens'].append(dict(b=s, c=c, n=len(rs), prompt=med(pt), agent=med([r['llm_prompt_tokens'] for r in rs]),
                comp_runs=sum(r['compactions'] > 0 for r in rs), comps=sum(r['compactions'] for r in rs),
                capped=sum(r.get('outputs_capped') or 0 for r in rs), wall=med([r['wall_s'] for r in rs]),
                jev=med([r['jev_input_tokens'] for r in rs]), turns=med([r['turns'] for r in rs]),
                maxturns=sum(r['end'] == 'max turns' for r in rs)))
            wa, to = tmean([r['wall_s'] for r in rs]), tmean(pt)
            for x in (wa, to):  # time and tokens cannot fall below 0
                if x['lo'] is not None:
                    x['lo'] = max(0.0, x['lo'])
            D['perrun'].append(dict(b=s, c=c, n=len(rs),
                wall_mean=wa['mean'], wall_lo=wa['lo'], wall_hi=wa['hi'], wall_med=med([r['wall_s'] for r in rs]),
                tok_mean=to['mean'], tok_lo=to['lo'], tok_hi=to['hi'], tok_sd=to['sd'], tok_med=med(pt),
                over=sum(r['over'] for r in rs), over_runs=sum(r['over'] > 0 for r in rs), comps=sum(r['compactions'] for r in rs),
                comp_runs=sum(r['compactions'] > 0 for r in rs), calls=sum(r['llm_calls'] for r in rs)))
            # attacks and utility under attack
            pl = pick(R, c, s, 'planted')
            harm = [bool(x['harm']) for x in pl if x['harm'] is not None]
            row = dict(b=s, c=c, runs=len(pl), harm=wilson(sum(harm), len(harm)), reached=wilson(sum(x['reached'] for x in pl), len(pl)))
            if c != 'plain':
                for f in ('harm', 'reached'):
                    # a paired binary outcome: newcombe's paired interval (method 10). a t interval on 1/0
                    # differences has no width when no task changed
                    pr, dr = paired(by_item(R, c, s, 'planted'), by_item(R, 'plain', s, 'planted'),
                                    lambda x, y, f=f: None if x[f] is None or y[f] is None else (bool(x[f]), bool(y[f])))
                    b_ = sum(x and y for x, y in pr); o1 = sum(x and not y for x, y in pr); o2 = sum(y and not x for x, y in pr)
                    d_, lo_, hi_ = newcombe_paired(b_, o1, o2, len(pr) - b_ - o1 - o2) if pr else (None, None, None)
                    row['d_' + f] = dict(n=len(pr), mean=d_, sd=None, lo=lo_, hi=hi_, dropped=dr, up=o1, same=len(pr) - o1 - o2, down=o2, newcombe=True)
            D['rates'].append(row)
            pc, dr = paired({k[:2]: v for k, v in by_item(R, c, s, 'planted').items()},
                            {k[:2]: v for k, v in by_item(R, c, s, 'clean').items()}, diff('score'))
            D['util'].append(dict(b=s, c=c, planted=tmean([x['score'] for x in pl]), runs=len(pl),
                                  minus_clean=dict(tmean(pc), dropped=dr, **updown(pc))))
    for a, ref in PAIRS:
        for s in SC:
            ds, dr = paired(by_item(R, a, s, 'clean'), by_item(R, ref, s, 'clean'), diff('score'))
            D['paired'].append(dict(c=a, ref=ref, b=s, **tmean(ds), dropped=dr, **updown(ds)))
    # wall time per task against plain, pooled, all runs and clean runs
    D['pwall'] = []
    for c in CF[1:]:
        for v in ('both', 'clean'):
            A, P = by_item(R, c, 'all', v), by_item(R, 'plain', 'all', v)
            ds, dr = paired(A, P, diff('wall_s'))
            ks = [k for k in sorted(set(A) & set(P)) if A[k]['wall_s'] is not None and P[k]['wall_s'] is not None]
            mc = st.mean([A[k]['wall_s'] for k in ks]) if ks else None
            mp = st.mean([P[k]['wall_s'] for k in ks]) if ks else None
            D['pwall'].append(dict(c=c, b='all', v=v, **tmean(ds), dropped=dr, **updown(ds), mean_c=mc, mean_p=mp,
                                   change=mc / mp - 1 if mp else None))
    for c in CF[1:]:
        for s in SC:
            row = dict(c=c, b=s)
            for v in VERSIONS:
                A, P = by_item(R, c, s, v), by_item(R, 'plain', s, v)
                rd, dr = paired(A, P, red)
                df, _ = paired(A, P, diff('total_prompt_tokens'))
                # the median per-task cut, and the tasks with the lowest cuts (the largest rises), which can drive the mean
                kr = sorted(((red(A[k], P[k]), A[k]['t']) for k in set(A) & set(P) if red(A[k], P[k]) is not None))
                row[v] = dict(red=dict(tmean(rd), med=med(rd)), diff=tmean(df), dropped=dr, low=[dict(t=t, red=x) for x, t in kr[:3]])
            D['ptok'].append(row)
    for c in CF:
        rs = pick(R, c)
        jt = sum(r['jev_input_tokens'] for r in rs)
        fb = lambda r: sum((r.get('jev_fallbacks') or {}).values())
        D['config'].append(dict(c=c, runs=len(rs), jev_tok=jt,
            prompt_tok=sum(r.get('total_prompt_tokens') or 0 for r in rs),
            call_prompt=sum(r.get('llm_prompt_tokens') or 0 for r in rs) / max(1, sum(r.get('llm_calls') or 0 for r in rs)), compl_tok=sum(r.get('total_completion_tokens') or 0 for r in rs),
            wall=med([r['wall_s'] for r in rs]), wallmax=mx([r['wall_s'] for r in rs]),
            jev_req=sum(r['jev_requests'] for r in rs), jev_err=sum(r['jev_errors'] for r in rs), jev_retry=sum(r['jev_retries'] for r in rs),
            fallbacks=sum(fb(r) for r in rs), stuck=sum(r.get('injection_stuck') or 0 for r in rs),
            llm_retries=sum(r['llm_retries'] for r in rs), comp_runs=sum(r['compactions'] > 0 for r in rs), comps=sum(r['compactions'] for r in rs),
            crashes=sum(r['c'] == c for r in F), reruns=sum(r['_reruns'] for r in rs),
            maxturns=sum(r['end'] == 'max turns' for r in rs)))
        at = pick(R, c, 'all', 'planted'); u = tmean([r['score'] for r in at])
        D['planted'].append(dict(c=c, harm=sum(bool(r['harm']) for r in at), n=len(at), seen=sum(r['reached'] for r in at),
                                 mean=u['mean'], lo=u['lo'], hi=u['hi']))
    return D


# agentdojo: the stock agent (defense none) against the jev injection filter, paired per case. defaults are gemma 4
# 31b's agentdojo run; a config's agentdojo block names its own root, folders and arms
AD_ROOT = str(WT / 'experiments' / 'runs-agentdojo-gemma4-31b')
AD_FOLDERS = {'banking': 'banking', 'slack': 'slack', 'travel': 'travel-yaml', 'workspace': 'workspace'}
AD_ARMS = {'stock': 'none', 'filter': 'jev-injection'}


def newcombe_paired(both, only1, only2, neither):
    # newcombe 1998 method 10: p1 - p2 for paired binary outcomes, wilson limits combined with the phi correlation
    n = both + only1 + only2 + neither
    p1, p2 = (both + only1) / n, (both + only2) / n
    w1, w2 = wilson(both + only1, n), wilson(both + only2, n)
    l1, u1, l2, u2 = w1['lo'], w1['hi'], w2['lo'], w2['hi']
    A = (both + only1) * (only2 + neither) * (both + only2) * (only1 + neither)
    # phi with newcombe's adjustment: ad - bc shrunk by n/2 when positive, 0 when that leaves it below 0
    ad_bc = both * neither - only1 * only2
    num = ad_bc - n / 2 if ad_bc > n / 2 else (0.0 if ad_bc >= 0 else ad_bc)
    phi = num / math.sqrt(A) if A else 0.0
    d = p1 - p2
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return d, max(-1.0, d - dl), min(1.0, d + du)


def agentdojo(root, folders=AD_FOLDERS, arms=AD_ARMS):
    # last record per episode key stands, a rerun record is not a result (as summarize.py)
    cases = defaultdict(dict)  # (suite, user task, injection task) -> arm -> record
    for s, f in folders.items():
        last = {}
        for r in jl(Path(root) / f / 'records.jsonl'):
            last[(r['suite'], r['user_task'], r['injection_task'] or 'none', r['attack'], r['defense_id'])] = r
        for k, r in last.items():
            if r.get('rerun'):
                continue
            arm = next((a for a, d in arms.items() if d == k[4]), None)
            if arm:
                cases[k[:4]][arm] = r
    measures = {'asr': ('attack', lambda r: r['security'] is True), 'uatk': ('attack', lambda r: r['utility'] is True),
                'una': ('none', lambda r: r['utility'] is True)}
    out = []
    for sc in list(folders) + ['overall']:
        for m, (kind, f) in measures.items():
            ks = [k for k in cases if (sc == 'overall' or k[0] == sc) and ((k[3] == 'none') == (kind == 'none'))]
            pairs = [(f(cases[k]['stock']), f(cases[k]['filter'])) for k in ks if len(cases[k]) == 2]
            n = len(pairs)
            both = sum(a and b for a, b in pairs)
            only_s = sum(a and not b for a, b in pairs)
            only_f = sum(b and not a for a, b in pairs)
            neither = n - both - only_s - only_f
            d, lo, hi = newcombe_paired(both, only_f, only_s, neither)  # filter minus stock
            t = tmean([int(b) - int(a) for a, b in pairs])
            out.append(dict(scope=sc, m=m, n=n, unpaired=len(ks) - n,
                            stock=wilson(both + only_s, n), filter=wilson(both + only_f, n),
                            both=both, only_stock=only_s, only_filter=only_f, neither=neither,
                            diff=d, lo=lo, hi=hi, t_lo=t['lo'], t_hi=t['hi']))
    recs = [r for c in cases.values() for r in c.values()]
    vals = lambda f: sorted({json.dumps(f(r), sort_keys=True) for r in recs})
    setup = dict(sampling=[json.loads(x) for x in vals(lambda r: r['sampling'])], providers=[json.loads(x) for x in vals(lambda r: r['provider_pin'])],
                 versions=[json.loads(x) for x in vals(lambda r: r['benchmark_version'])], model=[json.loads(x) for x in vals(lambda r: r['model'])],
                 travel_format={arm: sorted({r.get('tool_format') or 'str' for k, c in cases.items() if k[0] == 'travel' for a, r in c.items() if a == arm}) for arm in arms})
    return dict(rows=out, setup=setup)


def resolve(root, p):
    return Path(p) if Path(p).is_absolute() else Path(root) / p


def run_set(cfg, root, b16k=None):
    """a config's manifests and its two sets of counted records, as rows"""
    rc = cfg['runs']
    full = root / 'experiments' / 'main'
    if not resolve(root, rc['manifest']).exists():
        raise SystemExit(f"{resolve(root, rc['manifest'])}: no such manifest yet (the config's runs.manifest)")
    mA = json.load(open(resolve(root, rc['manifest'])))
    tasks = {t['id']: t for t in json.load(open(full / 'tasks.json'))['tasks']}
    # a run with no 16K step names no 16K manifest
    pB = Path(b16k) if b16k else resolve(root, rc['b16k_manifest']) if rc.get('b16k_manifest') else None
    mB = json.load(open(pB)) if pB and pB.exists() else None
    for m, t in ((mA, rc.get('tag')), (mB, rc.get('b16k_tag'))):
        if m and t and m['tag'] != t:
            raise SystemExit(f"manifest tag {m['tag']} is not the config's {t}")
    return mA, mB, tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(CONFIG), help='the agent config, default configs/gemma4-31b.json')
    ap.add_argument('--root', help="the repo root the run sets are under, default the config's root, else this repo")
    ap.add_argument('--b16k-manifest', help="default: the config's 16K manifest, if it exists")
    ap.add_argument('--out', help='default: out-<config name>/data.json next to this file')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    rc = cfg['runs']
    out = Path(a.out) if a.out else HERE / f'out-{Path(a.config).stem}' / 'data.json'
    root = Path(a.root or rc.get('root') or WT)
    mA, mB, tasks = run_set(cfg, root, a.b16k_manifest)
    ex = cfg['expected']
    if len(jobs_of(mA)) + len(mA.get('reused') or []) != ex['total']:
        print(f"warning: the manifest has {len(jobs_of(mA)) + len(mA.get('reused') or [])} runs, the config expects {ex['total']}")
    SA, _, FA = load_set(mA, root)
    SB, _, FB = load_set(mB, root) if mB else ({}, None, {})
    bA, bB = mA['compaction_budget'], (mB or {}).get('compaction_budget', 16384)
    RA, RB = rows_of(SA, bA, tasks), rows_of(SB, bB, tasks)
    D = compute(RA, rows_of(FA, bA, tasks))
    D['k16'] = compute(RB, rows_of(FB, bB, tasks)) if RB else None
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(D, open(out, 'w'))
    print(f'{out}: set A {len(RA)} runs, set B {len(RB)} runs')

if __name__ == '__main__':
    main()
