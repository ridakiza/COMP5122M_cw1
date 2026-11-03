"""Re-check every number used in Tasks 21-27 (group sizes, effective sample size,
Don't know by age, and the four "alternative tested" checks).

Run from the repo root:
    python scripts/check_task21_27.py

Reads the original SAV (raw Q30/Q31 codes incl. Don't know) and the shared cleaned
file (age groups, gender). Gap = % in 18-29 minus % in 65+, weighted unless stated,
Don't know/Refused kept in the total unless stated. Read-only.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
raw, meta = pyreadstat.read_sav(
    str(ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'),
    apply_value_formats=False, user_missing=True)
raw.columns = raw.columns.str.strip().str.lower()
clean = pd.read_csv(ROOT / 'data/clean/pew_clean_p1_provisional.csv', low_memory=False,
                    encoding='utf-8-sig')
assert raw.id.astype(int).tolist() == clean.id.astype(int).tolist()
labels = {k.lower(): v for k, v in meta.variable_value_labels.items()}
d = pd.DataFrame({
    'country': raw.country.astype(int).map({int(k): v for k, v in labels['country'].items()}),
    'w': raw.weight, 'q30': raw.ai_heard, 'q31': raw.ai_cncexc,
    'band': clean.age_band.astype('string').str.replace('–', '-', regex=False),
    'rec': clean.age_band_recovered.astype(str).str.lower() == 'true',
    'gender': clean.gender_clean})
NA = [8, 9]


def pct(g, col, code, weighted=True, dk='kept'):
    if dk == 'left out':
        g = g[~g[col].isin(NA)]
    w = g.w if weighted else pd.Series(1.0, index=g.index)
    return 100 * w[g[col] == code].sum() / w.sum()


def gaps(data, col, code, **kw):
    out = {}
    for c, g in data.groupby('country'):
        out[c] = pct(g[g.band == '18-29'], col, code, **kw) - pct(g[g.band == '65+'], col, code, **kw)
    return pd.Series(out)


def compare(name, a, b, show=1):
    ch = b - a
    top = ch.abs().sort_values(ascending=False).index[:show]
    tops = '; '.join(f'{c} {a[c]:.1f} -> {b[c]:.1f} ({ch[c]:+.1f})' for c in top)
    print(f'  {name:22s} average change {ch.abs().mean():.2f} | biggest: {tops}')
    return ch


ANS = [('q30', 1, 'Q30 "heard a lot"'), ('q31', 1, 'Q31 "more excited"'),
       ('q31', 2, 'Q31 "more concerned"'), ('q31', 3, 'Q31 "equally"')]

# ---------- Task 21 ----------
print('TASK 21 - people per country x age group')
cells = d.dropna(subset=['band']).groupby(['country', 'band']).agg(
    people=('w', 'size'), q30=('q30', lambda s: int((~s.isin(NA)).sum())),
    q31=('q31', lambda s: int((~s.isin(NA)).sum())))
print(f'  groups {len(cells)} | people with a group {int(cells.people.sum()):,} | '
      f'under 100/50/30: {(cells.people < 100).sum()}/{(cells.people < 50).sum()}/{(cells.people < 30).sum()}')
print('  under 100 (people, real Q30, real Q31):')
for (c, b), r in cells[cells.people < 100].sort_values('people').iterrows():
    print(f'    {c} {b}: {r.people}, {r.q30}, {r.q31}')
for c, b in (('Turkey', '65+'), ('India', '65+'), ('Israel', '50-64')):
    r = cells.loc[(c, b)]
    print(f'  Q31 no-answers {c} {b}: {r.people - r.q31} of {r.people}')
ind = d[(d.country == 'India') & (d.band == '18-29')]
print(f'  India 18-29: {len(ind)} ({int((~ind.rec).sum())} exact + {int(ind.rec.sum())} recovered)')

# ---------- Task 22 ----------
print('\nTASK 22 - Kish weight-only design effect')
k = d.groupby('country').w.agg(n='size', eff=lambda w: w.sum() ** 2 / (w ** 2).sum(),
                                lo='min', hi='max')
k['deff'] = k.n / k.eff
k = k.sort_values('deff', ascending=False)
for c in ('Italy', 'India', 'Australia'):
    r = k.loc[c]
    print(f'  {c}: deff {r.deff:.2f}, {int(r.n):,} people act like {r.eff:.0f} ({100 * r.eff / r.n:.0f}%)')
print('  highest:', ', '.join(f'{c} {v:.2f}' for c, v in k.deff.head(5).items()))
print('  lowest: ', ', '.join(f'{c} {v:.2f}' for c, v in k.deff.tail(4)[::-1].items()))
print(f'  Italy weights range {k.loc["Italy", "lo"]:.2f} to {k.loc["Italy", "hi"]:.2f}')
hy = d[(d.country == 'Hungary') & (d.band == '18-29')].w
print(f'  Hungary 18-29: {len(hy)} people act like {hy.sum() ** 2 / (hy ** 2).sum():.0f}')

# ---------- Task 23 ----------
print('\nTASK 23 - Q31 Don\'t know/Refused by age')
tk = d[d.country == 'Turkey']
for b in ('18-29', '30-49', '50-64', '65+'):
    g = tk[tk.band == b]
    na = g.q31.isin(NA)
    print(f'  Turkey {b}: {len(g)} people, DK {int((g.q31 == 8).sum())}, Ref '
          f'{int((g.q31 == 9).sum())}, no answer {100 * na.mean():.1f}% '
          f'(weighted {100 * g.w[na].sum() / g.w.sum():.1f}%)')
nr = {}
for c, g in d.groupby('country'):
    f = lambda b: 100 * g.w[(g.band == b) & g.q31.isin(NA)].sum() / g.w[g.band == b].sum()
    nr[c] = (f('18-29'), f('65+'))
nr = pd.DataFrame(nr, index=['young', 'old']).T
nr['diff'] = nr.old - nr.young
print('  weighted, 65+ minus 18-29:', ', '.join(
    f'{c} {r["diff"]:+.0f} ({r.young:.0f}% -> {r.old:.0f}%)' for c, r in nr.sort_values('diff', ascending=False).head(7).iterrows()))
print('  countries where older people give no answer more often:', int((nr['diff'] > 0).sum()), 'of 24;',
      'not:', list(nr.index[nr['diff'] <= 0]))

# ---------- Tasks 24-27: the four alternatives ----------
print('\nTASK 24 - Don\'t know kept vs left out (weighted)')
for col, code, name in ANS:
    compare(name, gaps(d, col, code), gaps(d, col, code, dk='left out'))

print('\nTASK 25 - weights on vs off (Don\'t know kept)')
for col, code, name in ANS:
    ch = compare(name, gaps(d, col, code), gaps(d, col, code, weighted=False))
    if col == 'q30':
        a, b = gaps(d, col, code), gaps(d, col, code, weighted=False)
        print(f'    Q30 gap smaller without weights in {int((b.abs() < a.abs()).sum())} of 24; '
              f'average change {(b - a).mean():+.1f}')
for c in ('Greece', 'Italy'):
    g = d[(d.country == c) & (d.band == '65+')]
    print(f'  {c} 65+ "a lot": unweighted {pct(g, "q30", 1, weighted=False):.0f}% vs '
          f'weighted {pct(g, "q30", 1):.0f}%')

print('\nTASK 26 - with vs without the 111 recovered ages')
no_rec = d.copy()
no_rec.loc[no_rec.rec, 'band'] = pd.NA
print(f'  people without an age group: {int(d.band.isna().sum())} -> {int(no_rec.band.isna().sum())}')
changed = set()
for col, code, name in ANS:
    ch = compare(name, gaps(d, col, code), gaps(no_rec, col, code))
    changed |= set(ch.index[ch.abs() > 1e-9])
print(f'  countries where any gap changes: {len(changed)}')

print('\nTASK 27 - Australia "Other" gender kept vs set to missing')
oth = d[d.gender == 3]
print(f'  "Other": {len(oth)} | by age: {oth.band.value_counts().reindex(["18-29", "30-49", "50-64", "65+"], fill_value=0).to_dict()}')
alt = d.copy()
alt.loc[alt.gender == 3, 'gender'] = np.nan
for col, code, name in ANS:
    a, b = gaps(d, col, code), gaps(alt, col, code)
    print(f'  {name:22s} age-only gaps: largest change {float((b - a).abs().max()):.1f}')
print('  gender-recorded gaps (only people with a gender):')
for col, code, name in ANS:
    a = gaps(d[d.gender.notna()], col, code)
    b = gaps(alt[alt.gender.notna()], col, code)
    ch = (b - a)[(b - a).abs() > 1e-9]
    print(f'    {name:22s} countries changed: {list(ch.index)} | '
          + ', '.join(f'{a[c]:.1f} -> {b[c]:.1f} ({ch[c]:+.1f})' for c in ch.index))