"""Task 4: check every number in the decision-log table that the
check_task4_shortcut.py script does not already cover.

Run from the repo root:
    python scripts/check_task4_table.py

Read-only; nothing is changed or saved.
"""
import re
from pathlib import Path

import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
raw, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
raw.columns = raw.columns.str.strip().str.lower()
labels = {k.lower(): v for k, v in meta.variable_value_labels.items()}


def no_answer(label) -> bool:
    return bool(re.search(r"don.?t know|refused", str(label), re.I))


print('"Why" row: what code 8 means in each question')
print('  Q30 (ai_heard) 8      =', repr(labels['ai_heard'][8.0]))
print('  Q32 (aireg_us) 8      =', repr(labels['aireg_us'][8.0]))
print('  UK education 8        =', repr(labels['d_educ_uk'][8.0]))
print('  UK education 9        =', repr(labels['d_educ_uk'][9.0]))

print('\n"Effect" row: questions with a "no answer" code')
labelled = [c for c in raw.columns if c in labels]
with_no_answer = [c for c in labelled if any(no_answer(v) for v in labels[c].values())]
print(f'  columns in the file:            {raw.shape[1]}')
print(f'  columns with value labels:      {len(labelled)}')
print(f'  ...with a Don\'t know/Refused code: {len(with_no_answer)}')

print('\n"Effect" row: codes in the data that have no label (unexplained)')
for col in labelled:
    s = raw[col].dropna()
    if not len(s):
        continue
    known = {float(k) for k in labels[col]}
    extra = sorted(set(s.unique()) - known)
    # Real-number columns (exact age, household size, adults, Israel's years of
    # school) only label their special codes; their many other values are plain
    # numbers, not codes. A coded question has at most a stray value or two.
    if len(extra) > 3:
        print(f'  (skipped {col}: a real-number column, {len(extra)} plain values)')
        continue
    if extra:
        for v in extra:
            print(f'  {col}: value {v:g} appears {int((s == v).sum())} times, no label')

print('\nSAV built-in "missing" settings (expected none):')
print('  ', meta.missing_ranges or 'none')