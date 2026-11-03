"""Task 4 re-check: what would the shortcut "treat 8, 9, 98 and 99 as no answer
everywhere" wrongly delete in the 64 Part I columns?

Run from the repo root:
    python scripts/check_task4_shortcut.py

For every Part I column and every person, it counts values equal to 8, 9, 98
or 99 whose Pew label is NOT "Don't know" or "Refused" (i.e. real answers).
Nothing is changed or saved.
"""
import re
from pathlib import Path

import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'

# The 64 Part I columns (same list as Task 3)
EXACT = ['id', 'country', 'weight', 'ai_heard', 'ai_cncexc', 'age', 'age_australia',
         'gender', 'gender_australia', 'internetfreq']
AIREG = ['aireg_china', 'aireg_country', 'aireg_eu', 'aireg_us']

raw, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
raw.columns = raw.columns.str.strip().str.lower()
labels = {k.lower(): v for k, v in meta.variable_value_labels.items()}
part1 = EXACT + AIREG + sorted(c for c in raw.columns if c.startswith(('d_age_', 'd_educ_')))
print(f'Part I columns: {len(part1)} (expected 64); people: {len(raw):,}')
assert len(part1) == 64 and len(raw) == 28_333


def is_no_answer(label) -> bool:
    return bool(re.search(r"don.?t know|refused", str(label), re.I))


groups = {'Q32 "Not sure"': 0, 'education': 0, 'country': 0, 'other': 0}
print('\nReal answers coded 8/9/98/99 (label is not Don\'t know/Refused):')
for col in part1:
    for code in (8, 9, 98, 99):
        n = int((raw[col] == code).sum())
        if n == 0:
            continue
        label = labels.get(col, {}).get(float(code), '<no label>')
        if is_no_answer(label):
            continue
        print(f'  {col:22s} code {code:>2}  n={n:>5}  label={label!r}')
        if col == 'id':
            continue  # IDs 98/99 are just two people's ID numbers, not answers
        if col.startswith('aireg_'):
            groups['Q32 "Not sure"'] += n
        elif col.startswith('d_educ_'):
            groups['education'] += n
        elif col == 'country':
            groups['country'] += n
        else:
            groups['other'] += n

print('\nTotals the shortcut would wrongly delete:')
for k, v in groups.items():
    print(f'  {k:16s} {v:>6,}')
print(f'  {"TOTAL":16s} {sum(groups.values()):>6,}')

print('\nCodes the shortcut does NOT touch (real answers anyway):')
print('  internetfreq = 6:', int((raw.internetfreq == 6).sum()), labels['internetfreq'][6.0])
print('  age = 97:       ', int((raw.age == 97).sum()), labels['age'][97.0])