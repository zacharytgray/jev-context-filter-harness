"""the paper's Table 2, the data behind Figures 3 and 4, the cost ratios and break-even prices of Section 5.7 and our rows of Table 3, from the
shipped run records. csv only, no plots.

    uv run --with numpy python experiments/main/paper_tables.py [--out experiments/main/out]

needs the data unpacked (scripts/unpack_data.py). it runs the analysis loader (analysis/data.py) per agent model and
compaction threshold, the cross-model script (cross_model/xmodel.py, which writes its out.json and per_task.csv into the out folder)
and cost_ratio.py, in process. about a minute. numpy is only for cost_ratio.py and is not a project dependency, so
`--with numpy` adds it for this one command.

writes, into the out folder:
    t1_main.csv, t1_main.tex  Table 2: per model, compaction threshold and configuration the score (mean over all task versions,
                              clean and planted), the median agent prompt tokens per clean run, the attacks followed,
                              the runs that compacted and the compactions, and the Jev input tokens per run (all runs)
    score_abs.csv             each configuration's score per test bed (and pooled, bed all) over clean, planted and all
                              versions: mean, t interval, runs, runs that scored 0
    score_versions.csv        the same cells as paired differences from Control, a pair being the same task version
                              under both configurations, t interval
    f2_effects.csv            Figure 3: (a) score minus Control over all task versions, paired, t interval; (b) the cut in agent
                              prompt tokens, 1 - mean(configuration) / mean(Control) over clean task pairs, bootstrap
                              interval (xmodel.py); (c) wall time minus Control over all versions, t interval; (d) the
                              planted attacks followed, a count
    f3_reading.csv            Figure 4 (a): mean raw characters per clean Control run at 32K, by where they came from
    f3_tasks.csv              Figure 4 (b): per clean task, Control's raw characters and relevance's token difference
    f3_rho.csv                Spearman's rho between the two, per model, with its permutation p
    cost_ratio.csv            Section 5.7: Jev input tokens per agent prompt token saved, and the break-even prices
    t4_agentdojo.csv          our AgentDojo rows: utility without attack, utility under attack and attack success,
                              stock against the injection filter, per suite and overall, newcombe's paired interval
model keys: gemma (Gemma 4 31B), mistral (Mistral Small 3.2 24B), deepseek (DeepSeek V4.1 Flash). configuration plain
is the paper's Control.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'analysis'))
sys.path.insert(0, str(HERE / 'cross_model'))
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import xmodel as XM  # noqa: E402
import cost_ratio as CR  # noqa: E402

MODELS = {'gemma': 'gemma4-31b.json', 'mistral': 'mistral-small-3.2-24b.json', 'deepseek': 'deepseek-v4.1-flash.json'}
LABEL = {'gemma': 'Gemma 4 31B', 'mistral': 'Mistral Small 3.2 24B', 'deepseek': 'DeepSeek V4.1 Flash'}
CF = D.CF
NAME = {'plain': 'Control', 'relevance': 'Relevance', 'injection': 'Injection',
        'relevance-injection': 'Relevance + injection'}
BUDGETS = ['32K', '16K']
AGENTDOJO = {'gemma': 'runs-agentdojo-gemma4-31b', 'deepseek': 'runs-agentdojo-deepseek-v4.1-flash'}
CLS = [('read_whole', 'raw_read_whole'), ('read_ranged', 'raw_read_ranged'), ('bash_snippet', 'raw_bash_snippet'),
       ('bash_other', 'raw_bash_other')]


def write_csv(path, rows, cols):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in cols})


def find(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    raise KeyError(kw)


def measures(cfgf):
    """data.py's measures for one agent, as data.py writes them, and its counted rows: {'32K': (D, R), '16K': ...}"""
    cfg = json.load(open(HERE / 'analysis' / 'configs' / cfgf))
    mA, mB, tasks = D.run_set(cfg, ROOT)
    out = {}
    for b, m in (('32K', mA), ('16K', mB)):
        S, _, F = D.load_set(m, ROOT)
        R = D.rows_of(S, m['compaction_budget'], tasks)
        out[b] = (D.compute(R, D.rows_of(F, m['compaction_budget'], tasks)), R)
    return out


VERSIONS = [('clean', 'clean'), ('planted', 'planted'), ('all', 'both')]  # our name, data.py's kind


def score_versions(M):
    """score over clean, planted and all task versions, per model, budget, configuration and test bed (bed all is
    pooled): each configuration's own mean (abs) and its paired difference from plain (pairs). the clean cells equal
    data.py's score and paired, the planted means its util.planted"""
    pr, ar = [], []
    for m in MODELS:
        for b in BUDGETS:
            B, R = M[m][b]
            for bed in D.SC:
                for c in CF:
                    for v, kind in VERSIONS:
                        xs = [x['score'] for x in D.pick(R, c, bed, kind) if x['score'] is not None]
                        t = D.tmean(xs)
                        if v == 'clean':
                            r = find(B['score'], c=c, b=bed)
                            assert t['n'] == r['n'] and t['mean'] == r['mean'], (m, b, c, bed)
                        ar.append(dict(model=m, budget=b, config=c, bed=bed, version=v, estimate=t['mean'], lo=t['lo'],
                                       hi=t['hi'], n=t['n'], zeros=sum(x == 0 for x in xs)))
                        if c == 'plain':
                            continue
                        ds, _ = D.paired(D.by_item(R, c, bed, kind), D.by_item(R, 'plain', bed, kind), D.diff('score'))
                        t = D.tmean(ds)
                        if v == 'clean':
                            r = find(B['paired'], c=c, ref='plain', b=bed)
                            assert t['n'] == r['n'] and t['mean'] == r['mean'], (m, b, c, bed)
                        pr.append(dict(model=m, budget=b, config=c, bed=bed, version=v, estimate=t['mean'], lo=t['lo'],
                                       hi=t['hi'], n=t['n']))
    return pr, ar


def table1(M, AB):
    rows = []
    for m in MODELS:
        for b in BUDGETS:
            B = M[m][b][0]
            for c in CF:
                s = find(AB, model=m, budget=b, config=c, bed='all', version='all')
                t = find(B['tokens'], c=c, b='all')
                r, cf = find(B['rates'], c=c, b='all'), find(B['config'], c=c)
                rows.append(dict(model=m, budget=b, config=c, score=s['estimate'], score_n=s['n'], tok_median=t['prompt'],
                                 tok_n=t['n'], followed_k=r['harm']['k'], followed_n=r['harm']['n'],
                                 comp_runs=cf['comp_runs'], comps=cf['comps'], runs=cf['runs'], jev_tok=cf['jev_tok'],
                                 jev_tok_run=cf['jev_tok'] / cf['runs']))
    return rows


def effects(M, X, SV):
    out = []
    for m in MODELS:
        for b in BUDGETS:
            B = M[m][b][0]
            for c in CF[1:]:
                r = find(SV, model=m, budget=b, config=c, bed='all', version='all')
                out.append(dict(model=m, budget=b, config=c, panel='score', estimate=r['estimate'], lo=r['lo'],
                                hi=r['hi'], n=r['n'], unit='score points, all versions'))
            for c in CF[1:]:
                r = find(B['pwall'], c=c, b='all', v='both')
                out.append(dict(model=m, budget=b, config=c, panel='wall', estimate=r['mean'], lo=r['lo'], hi=r['hi'],
                                n=r['n'], unit='seconds per run'))
            for c in CF:
                h = find(B['rates'], c=c, b='all')['harm']
                out.append(dict(model=m, budget=b, config=c, panel='attacks', estimate=h['k'], k=h['k'], n=h['n'],
                                unit='planted versions followed'))
            for c in CF[1:]:
                r = X[m]['ratio_cut'][b][c]
                if r['n'] != find(B['paired'], c=c, ref='plain', b='all')['n']:
                    raise SystemExit(f'{m} {b} {c}: xmodel pairs {r["n"]} differ from data.py')
                out.append(dict(model=m, budget=b, config=c, panel='tokens', estimate=100 * r['cut'],
                                lo=100 * r['boot_lo'], hi=100 * r['boot_hi'], n=r['n'], unit='% of control mean'))
    return out


def reading(X, xdir):
    cls_rows, task_rows, rho_rows = [], [], []
    with open(xdir / 'per_task.csv') as fh:
        pt = list(csv.DictReader(fh))
    for m in MODELS:
        P = X[m]['plain_per_run']['clean']
        for cls, f in CLS:
            cls_rows.append(dict(model=m, budget='32K', config='plain', cls=cls, mean_chars=P[f]['mean'], runs=P[f]['n']))
        for r in pt:
            if r['model'] == m:
                task_rows.append(dict(model=m, budget='32K', config='relevance', task=r['task'], testbed=r['testbed'],
                                      plain_raw_chars=int(r['plain_raw_chars']), plain_tokens=int(r['plain_tokens']),
                                      rel_tokens=int(r['rel_tokens']), diff=int(r['diff'])))
        sp = X[m]['spearman_raw_vs_diff']
        rho_rows.append(dict(model=m, budget='32K', config='relevance', rho=sp['rho'], p_perm=sp['p_perm'], n=sp['n']))
    return cls_rows, task_rows, rho_rows


def agentdojo():
    rows = []
    for m, folder in AGENTDOJO.items():
        for r in D.agentdojo(ROOT / 'experiments' / folder)['rows']:
            st, fl = r['stock'], r['filter']
            rows.append(dict(model=m, scope=r['scope'], measure=r['m'], n=r['n'], stock_k=st['k'],
                             stock_pct=round(100 * st['k'] / st['n'], 1), filter_k=fl['k'],
                             filter_pct=round(100 * fl['k'] / fl['n'], 1), diff=r['diff'], lo=r['lo'], hi=r['hi']))
    return rows


def write_t1(rows, path):
    """Table 2 as the paper sets it: three model blocks, 32K and 16K side by side. bold: the best value per model,
    compaction threshold and column (highest score, lowest of the rest), ties included; jev tokens are not bolded"""
    L = ['\\begin{table*}[t]',
         '\\caption{Absolute numbers. Score and Jev tokens (means) and compactions are over all runs, prompt tokens over '
         'clean runs, and attacks over planted runs. Bold: best per model and column, except Jev tokens. Differences are '
         'not tested here.}',
         '\\label{tab:main}',
         '\\footnotesize\\setlength{\\tabcolsep}{4pt}',
         '\\begin{tabular}{@{}l rrrrr @{\\hspace{12pt}} rrrrr@{}}',
         '\\toprule',
         ' & \\multicolumn{5}{c}{32K compaction threshold} & \\multicolumn{5}{c@{}}{16K compaction threshold} \\\\',
         '\\cmidrule(r{6pt}){2-6}\\cmidrule(l){7-11}']
    h1 = ' & Prompt tokens, & Attacks & Compactions: & Jev tokens'
    h2 = 'Score & median & followed & runs / total & per run'
    L += [f' & {h1} & {h1} \\\\', f'Configuration & {h2} & {h2} \\\\', '\\midrule']
    for mi, m in enumerate(MODELS):
        if mi:
            L.append('\\addlinespace[2pt]')
        L.append(f'\\multicolumn{{11}}{{@{{}}l}}{{\\emph{{{LABEL[m]}}}}} \\\\')
        best = {}
        for b in BUDGETS:
            blk = [r for r in rows if r['model'] == m and r['budget'] == b]
            best[b] = dict(score=max(r['score'] for r in blk), tok=min(r['tok_median'] for r in blk),
                           fol=min(r['followed_k'] for r in blk), comp=min(r['comp_runs'] for r in blk))
        bf = lambda on, x: f'\\textbf{{{x}}}' if on else x
        for c in CF:
            cells = []
            for b in BUDGETS:
                r = next(r for r in rows if r['model'] == m and r['budget'] == b and r['config'] == c)
                k = best[b]
                jv = r['jev_tok_run']
                cells.append(bf(r['score'] == k['score'], f'{r["score"]:.3f}') + ' & '
                             + bf(r['tok_median'] == k['tok'], f'{r["tok_median"]:,.0f}') + ' & '
                             + bf(r['followed_k'] == k['fol'], f'{int(r["followed_k"])}/{int(r["followed_n"])}') + ' & '
                             + bf(r['comp_runs'] == k['comp'], f'{int(r["comp_runs"])} / {int(r["comps"])}') + ' & '
                             + (f'{jv / 1000:,.0f}K' if jv else '0'))
            L.append(f'\\quad {NAME[c]} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
    path.write_text('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, default=HERE / 'out', help='output folder, default experiments/main/out')
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    M = {m: measures(f) for m, f in MODELS.items()}
    xdir = out / 'cross_model'
    XM.main(xdir)
    X = json.load(open(xdir / 'out.json'))
    SV, AB = score_versions(M)
    cols = ['model', 'budget', 'config', 'bed', 'version', 'estimate', 'lo', 'hi', 'n']
    write_csv(out / 'score_versions.csv', SV, cols)
    write_csv(out / 'score_abs.csv', AB, cols + ['zeros'])
    t1 = table1(M, AB)
    write_csv(out / 't1_main.csv', t1, ['model', 'budget', 'config', 'score', 'score_n', 'tok_median', 'tok_n',
                                        'followed_k', 'followed_n', 'comp_runs', 'comps', 'runs', 'jev_tok',
                                        'jev_tok_run'])
    write_t1(t1, out / 't1_main.tex')
    write_csv(out / 'f2_effects.csv', effects(M, X, SV),
              ['model', 'budget', 'config', 'panel', 'estimate', 'lo', 'hi', 'k', 'n', 'unit'])
    cls_rows, task_rows, rho_rows = reading(X, xdir)
    write_csv(out / 'f3_reading.csv', cls_rows, ['model', 'budget', 'config', 'cls', 'mean_chars', 'runs'])
    write_csv(out / 'f3_tasks.csv', task_rows, ['model', 'budget', 'config', 'task', 'testbed', 'plain_raw_chars',
                                                'plain_tokens', 'rel_tokens', 'diff'])
    write_csv(out / 'f3_rho.csv', rho_rows, ['model', 'budget', 'config', 'rho', 'p_perm', 'n'])
    CR.write(CR.compute(), out / 'cost_ratio.csv')
    write_csv(out / 't4_agentdojo.csv', agentdojo(), ['model', 'scope', 'measure', 'n', 'stock_k', 'stock_pct',
                                                      'filter_k', 'filter_pct', 'diff', 'lo', 'hi'])
    print(f'wrote {out}: t1_main.csv, t1_main.tex, score_abs.csv, score_versions.csv, f2_effects.csv, f3_reading.csv, f3_tasks.csv, f3_rho.csv, '
          f'cost_ratio.csv, t4_agentdojo.csv')


if __name__ == '__main__':
    main()
