"""Re-check every number used in Tasks 16-20 (age groups, gender, the full
cleaning check, Pew's 192 published figures, weights on vs off).

Run from the repo root:
    python scripts/check_task16_20.py

Reads the original SAV, the shared cleaned file and references/topline_expected.csv
(Pew's published figures). Read-only.
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
cname = raw.country.astype(int).map({int(k): v for k, v in labels['country'].items()})
AUS = raw.country == 13
band = clean.age_band.astype('string').str.replace('–', '-', regex=False)
GROUPS = ['18-29', '30-49', '50-64', '65+']

# ---------- Task 16: age groups ----------
print('TASK 16 - one age-group column')
t = pd.DataFrame({'outside AUS': band[~AUS].value_counts(dropna=False),
                  'Australia': band[AUS].value_counts(dropna=False)}).fillna(0).astype(int)
t['total'] = t.sum(axis=1)
print(t.reindex(GROUPS + [pd.NA]).rename(index={pd.NA: 'no group'}).to_string())
src = clean.age_band_source.value_counts(dropna=False)
print('  where each group came from:', src.to_dict())
ct = pd.crosstab(cname, band)
print(f'  smallest 18-29: {ct["18-29"].nsmallest(2).to_dict()} | largest: '
      f'{ct["18-29"].nlargest(1).to_dict()}')
for c in ('Kenya', 'Nigeria'):
    print(f'  {c}: 18-29 {ct.loc[c, "18-29"]} of {int((cname == c).sum()):,}')
print(f'  Netherlands 65+: {ct.loc["Netherlands", "65+"]} of {int((cname == "Netherlands").sum())}')
nog = cname[band.isna()].value_counts()
print(f'  most with no group: {nog.index[0]} {nog.iloc[0]} '
      f'({100 * nog.iloc[0] / (cname == nog.index[0]).sum():.1f}%)')
print(f'  people aged exactly 29 (outside Australia): {int((clean.age == 29).sum())}')

# ---------- Task 17: gender ----------
print('\nTASK 17 - one gender column')
print('  labels: gender', labels['gender'], '| gender_australia', labels['gender_australia'])
print('  before: gender', raw.gender.value_counts().to_dict(),
      '| gender_australia', raw.gender_australia.value_counts(dropna=False).to_dict())
print('  after (gender_clean):', clean.gender_clean.value_counts(dropna=False).to_dict())
g = pd.crosstab(cname, clean.gender_label)
for c in ('Japan', 'South Korea', 'South Africa'):
    print(f'  {c}: Male {g.loc[c, "Male"]}, Female {g.loc[c, "Female"]}')

# ---------- Task 18: everything together ----------
print('\nTASK 18 - the whole cleaned file')
print(f'  people kept: {len(clean):,} | columns: {clean.shape[1]}')
for col, codes in (('ai_heard', [8, 9]), ('ai_cncexc', [8, 9]), ('aireg_us', [9]),
                   ('aireg_china', [9]), ('aireg_eu', [9]), ('aireg_country', [9]),
                   ('internetfreq', [8, 9]), ('age', [98, 99])):
    print(f'  {col:14s} no-answer codes left: {int(clean[col].isin(codes).sum())}')
educ = [c for c in raw.columns if c.startswith('d_educ_')]
dk = pd.Series(False, index=raw.index)
for c in educ:
    nr = [k for k, v in labels.get(c, {}).items()
          if any(w in str(v).lower() for w in ("don't know", 'don’t know', 'refused'))]
    dk |= raw[c].isin(nr)
print(f'  education: people with Don\'t know/Refused still in raw columns: {int(dk.sum())} | '
      f'by country: {cname[dk].value_counts().head(7).to_dict()}')
print(f'  ... of which in Israel\'s years column: '
      f'{int(raw.d_educ_israel_years.isin([98, 99]).sum())}')

# ---------- Tasks 19 + 20: Pew's 192 published figures ----------
print('\nTASKS 19-20 - Pew\'s published Q30/Q31 figures (Don\'t know/Refused in the base)')
ref = pd.read_csv(ROOT / 'references/topline_expected.csv')
rows = []
for (q, c), grp in ref.groupby(['question', 'country']):
    m = cname == c
    x, w = raw.loc[m, q], raw.loc[m, 'weight']
    for _, r in grp.iterrows():
        hit = x.isin([8, 9]) if r.code == 8 else (x == r.code)
        rows.append({'q': q, 'country': c, 'code': r.code, 'pew': r.expected,
                     'weighted': 100 * w[hit].sum() / w.sum(),
                     'unweighted': 100 * hit.mean()})
r = pd.DataFrame(rows)
rnd = lambda s: np.floor(s + 0.5)                     # round half up, as published
print(f'  cells: {len(r)} | weighted match Pew: {int((rnd(r.weighted) == r.pew).sum())} | '
      f'unweighted match Pew: {int((rnd(r.unweighted) == r.pew).sum())}')
uk = r[(r.country == 'UK') & (r.q == 'ai_heard') & (r.code == 1)].iloc[0]
print(f'  UK Q30 "a lot": weighted {uk.weighted:.2f}% unweighted {uk.unweighted:.2f}% '
      f'Pew {uk.pew}% | diff {uk.unweighted - uk.weighted:+.2f}')
r['shift'] = r.weighted - r.unweighted
names = {('ai_heard', 1): 'Q30 a lot', ('ai_heard', 3): 'Q30 nothing at all',
         ('ai_heard', 8): 'Q30 DK/Ref', ('ai_cncexc', 2): 'Q31 more concerned',
         ('ai_cncexc', 8): 'Q31 DK/Ref'}
print('  biggest shifts (unweighted -> weighted):')
for _, s in r.reindex(r['shift'].abs().sort_values(ascending=False).index).head(6).iterrows():
    print(f'    {s.country} {names.get((s.q, s.code), f"{s.q} {s.code}")}: '
          f'{s.unweighted:.1f}% -> {s.weighted:.1f}%')
q30 = r[r.q == 'ai_heard']
alot = q30[q30.code == 1]; nothing = q30[q30.code == 3]
print(f'  Q30 "a lot" higher unweighted in {int((alot.unweighted > alot.weighted).sum())} of 24; '
      f'"nothing at all" lower unweighted in {int((nothing.unweighted < nothing.weighted).sum())} of 24')