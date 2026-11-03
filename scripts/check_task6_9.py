"""Re-check every number used in Tasks 6-9 (duplicates, unlabelled codes,
who-was-asked-what, weights), straight from the original SAV.

Run from the repo root:
    python scripts/check_task6_9.py

Read-only; nothing is changed or saved.
"""
from pathlib import Path

import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
raw, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
raw.columns = raw.columns.str.strip().str.lower()
labels = {k.lower(): v for k, v in meta.variable_value_labels.items()}
names = {int(k): v for k, v in labels['country'].items()}
AUS = raw.country == 13
print(f'{len(raw):,} people, {raw.shape[1]} columns, Australians {int(AUS.sum()):,}')

# ---------- Task 6: duplicates ----------
print('\nTASK 6 - duplicates')
dates = [c for c in raw.columns if c.startswith(('qdate', 'intdate'))]
print('  blank IDs:                       ', int(raw.id.isna().sum()))
print('  same ID twice:                   ', int(raw.id.duplicated().sum()))
print('  identical rows (all 483):        ', int(raw.duplicated().sum()))
print('  identical apart from ID (482):   ', int(raw.drop(columns='id').duplicated().sum()))
answers_only = raw.drop(columns=['id', 'weight'] + dates)
print(f'  same answers only ({answers_only.shape[1]} cols, no ID/weight/dates):',
      int(answers_only.duplicated().sum()))

# ---------- Task 7: codes with no label ----------
print('\nTASK 7 - codes the codebook does not explain')
print('  columns with labels:', sum(c in labels for c in raw.columns), 'of', raw.shape[1])
print('  columns with no labels:', [c for c in raw.columns if c not in labels])
for col in raw.columns:
    if col not in labels:
        continue
    extra = sorted(set(raw[col].dropna().unique()) - {float(k) for k in labels[col]})
    if 0 < len(extra) <= 3:          # >3 = a typed-in number column (age, household size...)
        for v in extra:
            print(f'  UNLABELLED: {col} = {v:g}, {int((raw[col] == v).sum())} people')
print('  d_income2_france labels:', sorted(int(k) for k in labels['d_income2_france']))

# ---------- Task 8: who was asked what ----------
print('\nTASK 8 - who was asked what (before any cleaning)')
for col in ('ai_heard', 'ai_cncexc', 'aireg_us', 'aireg_china', 'aireg_eu', 'aireg_country'):
    print(f'  {col:14s} answered by {int(raw[col].notna().sum()):,} of {len(raw):,}')
for col in ('age', 'gender', 'internetfreq'):
    print(f'  {col:14s} outside Australia {int(raw.loc[~AUS, col].notna().sum()):,} | '
          f'in Australia {int(raw.loc[AUS, col].notna().sum())}')
aus_cols = [c for c in raw.columns if c.endswith('_australia')]
outside = {c: int(raw.loc[~AUS, c].notna().sum()) for c in aus_cols}
print(f'  {len(aus_cols)} Australia-only columns, answers outside Australia: {sum(outside.values())}')
for c in ('age_australia', 'gender_australia', 'd_educ_australia'):
    print(f'  real gaps (Australians asked, no answer): {c} = {int(raw.loc[AUS, c].isna().sum())}')

# country-specific d_age_ / d_educ_ columns: answers only from their own country
key = {n.lower().replace(' ', ''): k for k, n in names.items()}
key.update({'safrica': key['southafrica'], 'skorea': key['southkorea']})
own = [c for c in raw.columns if c.startswith(('d_age_', 'd_educ_')) and c != 'd_educ_australia']
bad = 0
for c in own:
    country_part = c.split('_', 2)[2].split('_')[0]
    code = key[country_part]
    bad += int(raw.loc[raw.country != code, c].notna().sum())
n_age = sum(c.startswith('d_age_') for c in own)
print(f'  country-specific columns: {len(own)} ({n_age} age, {len(own) - n_age} education); '
      f'answers from another country: {bad}')

# age follow-up asked only of people who refused their exact age (98/99)
d_age = [c for c in raw.columns if c.startswith('d_age_')]
has_fu = raw[d_age].notna().any(axis=1)
not_refused = has_fu & ~raw.age.isin([98, 99])
print(f'  people with an age follow-up: {int(has_fu.sum())}; '
      f'of those, gave a real exact age anyway: {int(not_refused.sum())} '
      f'({", ".join(names[int(c)] for c in raw.loc[not_refused, "country"])})')
print('  dropping everyone with no internet answer (incl. 8/9) would remove:',
      int(raw.internetfreq.isna().sum()), 'people,',
      int((raw.internetfreq.isna() & AUS).sum()), 'of them Australians')

# ---------- Task 9: weights ----------
print('\nTASK 9 - weights')
w = raw.weight
print('  missing:', int(w.isna().sum()), '| zero or negative:', int((w <= 0).sum()))
g = raw.groupby('country').agg(n=('weight', 'size'), total=('weight', 'sum'))
print(f'  countries where weights add up to the number of people: '
      f'{int(((g.total - g.n).abs() < 1e-6).sum())} of {len(g)}; '
      f'largest gap {float((g.total - g.n).abs().max()):.1e}')
print(f'  UK: {int(g.loc[12, "n"]):,} people, weights add to {g.loc[12, "total"]:.6f}')
lo, hi = w.idxmin(), w.idxmax()
print(f'  smallest weight {w[lo]:.4f} ({names[int(raw.country[lo])]}), '
      f'largest {w[hi]:.4f} ({names[int(raw.country[hi])]})')