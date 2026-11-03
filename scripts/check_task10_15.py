"""Re-check every number used in Tasks 10-15 (blanking Q30/Q31/Q32/internet/age,
recovering age groups, the Israeli conflict).

Run from the repo root:
    python scripts/check_task10_15.py

Reads the original SAV (before cleaning) and the shared cleaned file
data/clean/pew_clean_p1_provisional.csv (after cleaning). Read-only.
"""
from pathlib import Path

import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
raw, meta = pyreadstat.read_sav(
    str(ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'),
    apply_value_formats=False, user_missing=True)
raw.columns = raw.columns.str.strip().str.lower()
clean = pd.read_csv(ROOT / 'data/clean/pew_clean_p1_provisional.csv', low_memory=False,
                    encoding='utf-8-sig')
assert raw.id.astype(int).tolist() == clean.id.astype(int).tolist(), 'row order differs'
labels = {k.lower(): v for k, v in meta.variable_value_labels.items()}
names = {int(k): v for k, v in labels['country'].items()}
cname = raw.country.astype(int).map(names)
AUS = raw.country == 13


def by_country(mask, top=5):
    s = cname[mask].value_counts()
    return ', '.join(f'{k} {v}' for k, v in s.head(top).items())


# ---------- Task 10: Q30 / Q31 ----------
print('TASK 10 - Q30 and Q31: Don\'t know (8) and Refused (9) blanked')
for col in ('ai_heard', 'ai_cncexc'):
    dk, ref = int((raw[col] == 8).sum()), int((raw[col] == 9).sum())
    print(f'  {col}: before {raw[col].notna().sum():,} | blanked {dk + ref:,} '
          f'({dk:,} DK + {ref} Refused) | real answers after {clean[col].notna().sum():,} '
          f'| left: {sorted(clean[col].dropna().unique().astype(int))}')
    print(f'      India: {int((raw[col].isin([8, 9]) & (cname == "India")).sum())} | '
          f'top countries: {by_country(raw[col].isin([8, 9]))}')
    print(f'      Australia: DK {int(((raw[col] == 8) & AUS).sum())}, '
          f'Refused {int(((raw[col] == 9) & AUS).sum())}')
print('  people removed:', len(raw) - len(clean))

# ---------- Task 11: Q32 ----------
print('\nTASK 11 - Q32: only Refused (9) blanked, "Not sure" (8) kept')
tot8 = tot9 = 0
for col in ('aireg_us', 'aireg_china', 'aireg_eu', 'aireg_country'):
    n8, n9 = int((raw[col] == 8).sum()), int((raw[col] == 9).sum())
    tot8 += n8; tot9 += n9
    print(f'  {col:14s} Not sure kept {n8:,} (after: {int((clean[col] == 8).sum()):,}) | '
          f'Refused blanked {n9} | valid after {clean[col].notna().sum():,}')
print(f'  totals: Not sure {tot8:,} | Refused {tot9:,}')
print(f'  EU refusals in India: {int(((raw.aireg_eu == 9) & (cname == "India")).sum())} of '
      f'{int((raw.aireg_eu == 9).sum())} | Japan "Not sure" on EU: '
      f'{int(((raw.aireg_eu == 8) & (cname == "Japan")).sum())}')
q32 = ['aireg_us', 'aireg_china', 'aireg_eu', 'aireg_country']
for c in ('Australia', 'Japan'):
    print(f'  {c} refusals on Q32: {int((raw.loc[cname == c, q32] == 9).sum().sum())}')

# ---------- Task 12: internet use ----------
print('\nTASK 12 - internet use: 8 and 9 blanked, 6 (non-user) kept')
i = raw.internetfreq
print(f'  recorded before {i.notna().sum():,} | blanked {int(i.isin([8, 9]).sum())} '
      f'(DK {int((i == 8).sum())}, Refused {int((i == 9).sum())}) | non-users kept '
      f'{int((clean.internetfreq == 6).sum()):,}')
print(f'  valid after {clean.internetfreq.notna().sum():,} | total missing '
      f'{clean.internetfreq.isna().sum():,}')
print(f'  non-users by country: {by_country(i == 6, 4)} | India blanked: '
      f'{int((i.isin([8, 9]) & (cname == "India")).sum())}')

# ---------- Task 13: exact age ----------
print('\nTASK 13 - exact age: 98/99 blanked, 97 ("97 or older") kept')
a = raw.age
print(f'  recorded before {a.notna().sum():,} | ages 18-96 {int(a.between(18, 96).sum()):,} | '
      f'97+ {int((a == 97).sum())} | blanked {int(a.isin([98, 99]).sum())} '
      f'(DK {int((a == 98).sum())}, Refused {int((a == 99).sum())})')
print(f'  usable after {clean.age.notna().sum():,} | missing {clean.age.isna().sum():,} | '
      f'age_australia present {raw.age_australia.notna().sum():,}')
print(f'  missing ages by country: {by_country(a.isin([98, 99]))}')

# ---------- Task 14: recovering age groups ----------
print('\nTASK 14 - age groups recovered from the follow-up')
refused = a.isin([98, 99])
rec = clean.age_band_recovered.astype(str).str.lower() == 'true'
d_age = [c for c in raw.columns if c.startswith('d_age_')]
fu = raw[d_age].bfill(axis=1).iloc[:, 0]                 # each person's follow-up code
fu_no_answer = pd.Series(False, index=raw.index)
for c in d_age:
    lab = labels.get(c, {})
    nr = [k for k, v in lab.items() if any(w in str(v).lower() for w in ("don't", 'don’t', 'refused'))]
    fu_no_answer |= raw[c].isin(nr)
print(f'  missing exact age (outside Australia): {int(refused.sum())} | with a follow-up '
      f'answer: {int((refused & raw[d_age].notna().any(axis=1)).sum())}')
band = clean.age_band.astype('string').str.replace('–', '-', regex=False)
print(f'  placed: {int(rec.sum())} -> {band[rec].value_counts().sort_index().to_dict()}')
left = refused & ~rec
print(f'  not placed: {int(left.sum())} = refused the follow-up too '
      f'{int((left & fu_no_answer).sum())} + band crosses two groups '
      f'{int((left & ~fu_no_answer).sum())}')
t = pd.DataFrame({'missing': cname[refused].value_counts(), 'placed': cname[rec].value_counts()}).fillna(0).astype(int)
t['left'] = t.missing - t.placed
print('  by country (missing -> placed -> left):')
print('   ', ' | '.join(f'{c} {r.missing}->{r.placed}->{r.left}'
                         for c, r in t.sort_values('missing', ascending=False).iterrows()))
print(f'  exact ages invented: {int((clean.age.notna() & a.isin([98, 99])).sum())} | '
      f'no age group at all: {int(band.isna().sum())}')

# ---------- Task 15: Israel ----------
print('\nTASK 15 - Israelis with an exact age AND a follow-up answer')
both = raw[d_age].notna().any(axis=1) & ~refused & raw.age.notna()
for idx in raw.index[both]:
    c = next(c for c in d_age if pd.notna(raw.at[idx, c]))
    code = raw.at[idx, c]
    print(f'  {cname[idx]}: id {int(raw.id[idx])}, exact age {int(a[idx])}, follow-up '
          f'{labels[c][code]!r}, age group used {band[idx]}')