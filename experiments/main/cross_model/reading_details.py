"""the reading and jev latency details the paper gives beside figure 4, at the 32K threshold, from the run traces.
uses the cross-model script's own call classes (xmodel.py) and the analysis loader's run selection (data.py).

    python3 experiments/main/cross_model/reading_details.py      # stdlib only; prints, writes nothing

per model, over the clean Control (plain) runs: runs with a whole-file read, the share of raw characters from
whole-file reads, runs with 10 or more ranged reads and the median line span of a ranged read. over the clean
relevance runs: outputs longer than 4,000 characters by class, and the share of their raw characters relevance kept.
over every relevance + injection run (clean and planted): the median latency of a jev request, send to parsed reply
at the client (trace.jsonl latency_ms).
"""
import statistics as st
from pathlib import Path

import xmodel as X

D = X.D


def spans(run):
    out = []
    for d in D.jl(Path(run) / 'trace.jsonl'):
        if d.get('event') == 'calls':
            for c in d['calls']:
                a = c.get('args') or {}
                if c.get('tool') == 'read' and (a.get('start_line') or a.get('end_line')):
                    try:
                        sl, el = int(a.get('start_line') or 1), int(a['end_line'])
                        out.append(el - sl + 1)
                    except (KeyError, TypeError, ValueError):
                        pass
    return out


def main():
    for key, name, cfgname in X.CONFIGS:
        _, RA, _, _ = X.load(cfgname)
        print(f'## {name} (32K)')
        plain = D.pick(RA, 'plain', 'all', 'clean')
        per = [X.run_stats(X.trace_calls(r['_run'])) for r in plain]
        raw = sum(p['raw_chars'] for p in per)
        whole = sum(p['raw_read_whole'] for p in per)
        print(f'  clean Control runs: {len(per)}; with a whole-file read: {sum(p["read_whole"] > 0 for p in per)}; '
              f'raw chars from whole-file reads: {100 * whole / raw:.1f}%')
        sp = [s for r in plain for s in spans(r['_run'])]
        print(f'  runs with 10 or more ranged reads: {sum(p["read_ranged"] + p["read_whole_by_range"] >= 10 for p in per)}; '
              f'median lines per ranged read: {st.median(sp) if sp else "-"} ({len(sp)} ranged reads)')
        cs = [c for r in D.pick(RA, 'relevance', 'all', 'clean') for c in X.trace_calls(r['_run']) if c['ran'] and c['ok']]
        big = [c for c in cs if not c['pass_through']]
        rd = [c for c in big if c['tool'] == 'read']
        kept = lambda xs: 100 * sum(c['entered'] or 0 for c in xs) / max(1, sum(c['raw'] for c in xs))
        print(f'  clean relevance outputs over 4,000 chars: {len(big)}; bash search: {sum(c["kind"] == "search" for c in big)}; '
              f'reads: {len(rd)}; chars kept: all {kept(big):.1f}%, reads {kept(rd):.1f}%')
        lat = [d['latency_ms'] for r in D.pick(RA, 'relevance-injection')
               for d in D.jl(Path(r['_run']) / 'trace.jsonl') if d.get('kind') == 'request' and d.get('latency_ms') is not None]
        print(f'  relevance + injection runs: {len(D.pick(RA, "relevance-injection"))}; jev requests {len(lat):,}; '
              f'median latency {st.median(lat):.0f} ms')


if __name__ == '__main__':
    main()
