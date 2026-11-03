"""Check the Task 9 'Effect' numbers: what changes if the weights are NOT used?

Run from the repo root:
    python scripts/check_weights_effect_final.py

Uses the shared cleaned file (data/clean/pew_clean_final.csv).
Q30 "A lot" = code 1. Don't know/Refused (now blank) stay in the base, as Pew does.
Read-only; nothing is changed or saved.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / 'data/clean/pew_clean_final.csv', low_memory=False,
                encoding='utf-8-sig')
d['age_band'] = d.age_band.astype('string').str.replace('–', '-', regex=False)
d['a_lot'] = (d.ai_heard == 1).astype(float)          # blank (DK/Refused) counts as "not a lot"


def pct(g, weighted):
    w = g.weight if weighted else pd.Series(1.0, index=g.index)
    return 100 * (w * g.a_lot).sum() / w.sum()


# 1. UK, Q30 "heard a lot about AI"
uk = d[d.country_name == 'UK']
print(f'UK Q30 "A lot": weighted {pct(uk, True):.1f}% | unweighted {pct(uk, False):.1f}% '
      f'| people {len(uk):,}, said "A lot" {int(uk.a_lot.sum())}')

# 2. Young-vs-old gap (18-29 minus 65+) per country, weighted vs unweighted
rows = []
for c, g in d.groupby('country_name'):
    y, o = g[g.age_band == '18-29'], g[g.age_band == '65+']
    gw = pct(y, True) - pct(o, True)
    gu = pct(y, False) - pct(o, False)
    rows.append({'country': c, 'gap_weighted': gw, 'gap_unweighted': gu, 'change': gu - gw})
t = pd.DataFrame(rows).sort_values('change', key=abs, ascending=False)
print(f'\nQ30 "A lot" gap, 18-29 minus 65+, {len(t)} countries')
print(f'  average size of the change when weights are dropped: {t.change.abs().mean():.2f} points')
print(f'  countries where the gap gets smaller without weights: '
      f'{int((t.gap_unweighted.abs() < t.gap_weighted.abs()).sum())} of {len(t)}')
print('  biggest changes:')
print(t.head(5).to_string(index=False, float_format=lambda x: f'{x:.1f}'))