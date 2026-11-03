"""Confirm the final dataset = provisional dataset + edu_level, so every Task 2-35
check still holds. Then re-run the check scripts on the final file.

Run from the repo root:
    python scripts/check_final_dataset.py
"""
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / 'data/clean/pew_clean_p1_provisional.csv'
FINAL = ROOT / 'data/clean/pew_clean_final.csv'

read = lambda p: pd.read_csv(p, low_memory=False, encoding='utf-8-sig', float_precision='round_trip')
p, f = read(PROV), read(FINAL)
print('STEP 1 - final vs provisional')
print(f'  rows {len(p):,} vs {len(f):,} | new columns {[c for c in f if c not in p]} | '
      f'dropped {[c for c in p if c not in f]}')
same = f[p.columns].equals(p)
print(f'  all {p.shape[1]} original columns identical, cell for cell: {same}')
print('  edu_level:', f['edu_level'].value_counts(dropna=False).to_dict())

# STEP 2 - re-run the check scripts that read the cleaned CSV, on both files
print('\nSTEP 2 - check scripts, provisional vs final (output should match)')
for name in ('check_task10_15', 'check_task16_20', 'check_task21_27', 'check_task31_33',
             'check_weights_effect'):
    src = ROOT / 'scripts' / f'{name}.py'
    if not src.exists():
        print(f'  {name}: not found, skipped'); continue
    tmp = ROOT / 'scripts' / f'_final_{name}.py'
    tmp.write_text(src.read_text(encoding='utf-8').replace('pew_clean_p1_provisional', 'pew_clean_final'),
                   encoding='utf-8')
    run = lambda s: subprocess.run([sys.executable, str(s)], capture_output=True, text=True, cwd=ROOT).stdout
    a, b = run(src), run(tmp)
    tmp.unlink()
    norm = lambda t: re.sub(r'columns: \d+', 'columns: N', t)
    print(f'  {name:22s} {"SAME" if norm(a) == norm(b) else "DIFFERENT - check"}')
    if name == 'check_task16_20':
        m = re.search(r'cells: (\d+) \| weighted match Pew: (\d+)', b)
        if m:
            print(f'      Pew topline on final file: {m.group(2)}/{m.group(1)} match')