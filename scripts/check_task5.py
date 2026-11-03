"""Task 5 re-check: do the SAV and CSV hold the same answers, and what goes
wrong if the CSV is loaded as-is?

Run from the repo root:
    python scripts/check_task5.py

Read-only; nothing is changed or saved.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'

sav, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
sav.columns = sav.columns.str.strip().str.lower()
csv = pd.read_csv(CSV, low_memory=False, encoding='utf-8-sig')       # loaded as-is
csv.columns = csv.columns.str.strip().str.lower()
print(f'SAV: {sav.shape[0]:,} people x {sav.shape[1]} columns')
print(f'CSV: {csv.shape[0]:,} people x {csv.shape[1]} columns')

# 1. The CSV loaded as-is: blank cells are stored as a single space
print('\n1. CSV loaded as-is')
spaces = {c: int((csv[c].astype(str) == ' ').sum()) for c in csv.columns}
print(f'   cells that are just a space " ": {sum(spaces.values()):,} in '
      f'{sum(v > 0 for v in spaces.values())} columns')
aus = sav.country == 13  # 13 = Australia
print(f'   Australians: {int(aus.sum()):,}')
for col in ('age', 'gender', 'internetfreq'):
    sav_answered = int(sav[col].notna().sum())
    csv_answered = int(csv[col].notna().sum())   # a space counts as "answered"
    text = not pd.api.types.is_numeric_dtype(csv[col])
    print(f'   {col:12s} SAV answered {sav_answered:>6,} | CSV as-is "answered" '
          f'{csv_answered:>6,} | fake extra {csv_answered - sav_answered:>5,} | '
          f'CSV loads as text: {text}')
n_text = sum((not pd.api.types.is_numeric_dtype(csv[c])) and pd.api.types.is_numeric_dtype(sav[c])
             for c in csv.columns)
print(f'   number columns that load as text in the CSV: {n_text}')

# 2. Fix the spaces, then compare every value, person by person
print('\n2. After turning spaces into blanks, compare every value')
fixed = csv.replace(r'^\s*$', np.nan, regex=True)
a = sav.set_index('id').sort_index()
b = fixed.set_index('id').sort_index()
assert a.index.equals(b.index.astype(a.index.dtype)), 'different people'
b.index = a.index
assert list(a.columns) == list(b.columns), 'different column names'
total_diff = 0
for col in a.columns:
    x, y = a[col], b[col]
    if not pd.api.types.is_numeric_dtype(x) or col.startswith(('qdate', 'intdate')):
        xs = x.astype(str).str.strip().replace({'nan': ''})
        ys = y.astype(str).str.strip().replace({'nan': ''})
        if col.startswith(('qdate', 'intdate')):        # compare dates as dates
            xs = pd.to_datetime(xs, errors='coerce'); ys = pd.to_datetime(ys, errors='coerce')
            d = int(((xs != ys) & ~(xs.isna() & ys.isna())).sum())
        else:
            d = int((xs != ys).sum())
    else:
        yn = pd.to_numeric(y, errors='coerce')
        d = int((~np.isclose(x.astype(float), yn.astype(float), equal_nan=True, rtol=0, atol=1e-9)).sum())
    if d:
        print(f'   {col}: {d} differences')
    total_diff += d
print(f'   columns compared: {a.shape[1] + 1} (incl. id) | people: {len(a):,} | '
      f'differences: {total_diff}')