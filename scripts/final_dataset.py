"""Build the FINAL cleaned dataset = P1's provisional file + edu_level (Low/Medium/High).

Run from the repo root:
    python scripts/make_final_dataset.py

Reads : data/clean/pew_clean_p1_provisional.csv
Writes: data/clean/pew_clean_final.csv   (never commit anything in data/clean - repo is public)

Mapping = P2's EDU_MAP. One switch below (ISCED_FIX) moves the three
"post-school, below college" codes (Greece 7, India 7, India 8) from High to
Medium so the rule is the same in every country. Set it once the team agrees.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ISCED_FIX = True          # True = Greece 7, India 7, India 8 -> Medium (agreed rule); False = P2's original

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data/clean/pew_clean_p1_provisional.csv'
OUT = ROOT / 'data/clean/pew_clean_final.csv'

# country code -> that country's education column (P2's mapping, unchanged)
COLUMN = {2: 'd_educ_canada', 3: 'd_educ_france', 4: 'd_educ_germany', 5: 'd_educ_greece',
          6: 'd_educ_hungary', 7: 'd_educ_italy', 8: 'd_educ_netherlands', 9: 'd_educ_poland',
          10: 'd_educ_spain', 11: 'd_educ_sweden', 12: 'd_educ_uk', 13: 'd_educ_australia',
          14: 'd_educ_india', 15: 'd_educ_indonesia', 16: 'd_educ_japan', 17: 'd_educ_skorea',
          18: 'd_educ_israel', 19: 'd_educ_turkey', 20: 'd_educ_kenya', 21: 'd_educ_nigeria',
          22: 'd_educ_safrica', 23: 'd_educ_argentina', 24: 'd_educ_brazil', 25: 'd_educ_mexico'}

# ranges END one past the last code (range(1, 6) = codes 1-5). Codes not listed
# (98 Don't know, 99 Refused, Germany 8/9) stay blank.
EDU_MAP = {
    'd_educ_canada':      {'Low': range(1, 6),  'Medium': range(6, 9),  'High': range(9, 12)},
    'd_educ_france':      {'Low': range(1, 4),  'Medium': range(4, 9),  'High': range(9, 14)},
    'd_educ_germany':     {'Low': range(1, 3),  'Medium': range(3, 5),  'High': range(5, 8)},
    'd_educ_greece':      {'Low': range(1, 5),  'Medium': range(5, 7),  'High': range(7, 13)},
    'd_educ_hungary':     {'Low': range(1, 4),  'Medium': range(4, 9),  'High': range(9, 12)},
    'd_educ_italy':       {'Low': range(1, 4),  'Medium': range(4, 6),  'High': range(6, 9)},
    'd_educ_netherlands': {'Low': range(1, 5),  'Medium': range(5, 8),  'High': range(8, 11)},
    'd_educ_poland':      {'Low': range(1, 4),  'Medium': range(4, 8),  'High': range(8, 11)},
    'd_educ_spain':       {'Low': range(1, 6),  'Medium': range(6, 8),  'High': range(8, 13)},
    'd_educ_sweden':      {'Low': range(1, 4),  'Medium': range(4, 6),  'High': range(6, 11)},
    'd_educ_uk':          {'Low': range(1, 3),  'Medium': range(3, 5),  'High': range(5, 10)},
    'd_educ_australia':   {'Low': range(1, 2),  'Medium': range(2, 3),  'High': range(3, 4)},
    'd_educ_india':       {'Low': range(1, 6),  'Medium': range(6, 7),  'High': range(7, 12)},
    'd_educ_indonesia':   {'Low': range(1, 7),  'Medium': range(7, 8),  'High': range(8, 13)},
    'd_educ_japan':       {'Low': range(1, 4),  'Medium': range(4, 5),  'High': range(5, 10)},
    'd_educ_skorea':      {'Low': range(1, 5),  'Medium': range(5, 7),  'High': range(7, 11)},
    'd_educ_israel':      {'Low': range(1, 5),  'Medium': [*range(5, 11), 12], 'High': [11, *range(13, 16)]},
    'd_educ_turkey':      {'Low': range(1, 7),  'Medium': range(7, 9),  'High': range(9, 12)},
    'd_educ_kenya':       {'Low': range(1, 7),  'Medium': range(7, 10), 'High': range(10, 14)},
    'd_educ_nigeria':     {'Low': range(1, 5),  'Medium': range(5, 7),  'High': range(7, 12)},
    'd_educ_safrica':     {'Low': range(1, 7),  'Medium': range(7, 12), 'High': range(12, 21)},
    'd_educ_argentina':   {'Low': range(1, 5),  'Medium': range(5, 8),  'High': range(8, 12)},
    'd_educ_brazil':      {'Low': range(1, 7),  'Medium': range(7, 9),  'High': range(9, 13)},
    'd_educ_mexico':      {'Low': range(1, 6),  'Medium': range(6, 10), 'High': range(10, 13)},
}
LOOKUP = {col: {c: lvl for lvl, codes in levels.items() for c in codes} for col, levels in EDU_MAP.items()}
if ISCED_FIX:
    LOOKUP['d_educ_greece'][7] = 'Medium'     # IEK: post-secondary, not tertiary
    LOOKUP['d_educ_india'][7] = 'Medium'      # technical/industrial training certificate
    LOOKUP['d_educ_india'][8] = 'Medium'      # professional training diploma (GNM, teacher training)

# ---------- build ----------
# round_trip keeps the weights exactly as they were (no 4e-16 drift on re-save)
df = pd.read_csv(SRC, low_memory=False, encoding='utf-8-sig', float_precision='round_trip')
before = df.copy()

edu = pd.Series(np.nan, index=df.index, dtype=object)
for code, col in COLUMN.items():
    m = df['country'] == code
    edu[m] = df.loc[m, col].map(LOOKUP[col])
# Germany code 3: vocational training yes -> Medium, no -> Low
ger3 = (df['country'] == 4) & (df['d_educ_germany'] == 3)
edu[ger3] = np.where(df.loc[ger3, 'd_educ_germany_vocational'] == 1, 'Medium', 'Low')
df['edu_level'] = edu

# ---------- checks (stop if anything is wrong) ----------
assert len(df) == len(before) == 28333, 'row count changed'
assert df['id'].equals(before['id']), 'row order / ids changed'
assert df.drop(columns='edu_level').equals(before), 'an existing column changed'
assert set(df['edu_level'].dropna()) == {'Low', 'Medium', 'High'}
for code, col in COLUMN.items():                       # every real answer got a level
    m = (df['country'] == code) & df[col].notna() & ~df[col].isin([8, 9, 98, 99] if code == 4 else [98, 99])
    assert df.loc[m, 'edu_level'].notna().all(), f'{col}: real answer left blank'
    assert df.loc[df['country'] == code, col].isin([98, 99]).sum() == 0 or \
        df.loc[(df['country'] == code) & df[col].isin([98, 99]), 'edu_level'].isna().all()

df.to_csv(OUT, index=False, encoding='utf-8-sig')
back = pd.read_csv(OUT, low_memory=False, encoding='utf-8-sig', float_precision='round_trip')
assert back['weight'].equals(df['weight']), 'weights changed on save'

print(f'ISCED_FIX = {ISCED_FIX}')
print('edu_level:', df['edu_level'].value_counts(dropna=False).to_dict())
print(f'Germany code 3 -> Medium {int((df.edu_level[ger3] == "Medium").sum())}, Low {int((df.edu_level[ger3] == "Low").sum())}')
print(f'rows {len(df):,} | columns {df.shape[1]} (was {before.shape[1]}) | all checks passed')
print(f'saved {OUT.relative_to(ROOT)}  - do NOT commit')