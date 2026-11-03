"""Compare our cleaned data with Pew's published Q30/Q31 figures, read straight from
Pew's Topline PDF (page 48) - no hand-typed numbers in between.

Run from the repo root (needs: pip install pypdf):
    python scripts/check_topline_pdf.py

Three-way check for every number (24 countries x Q30/Q31 x 4 answers = 192):
  1. PDF  vs references/topline_expected.csv  (was the CSV typed in correctly?)
  2. PDF  vs our data, WEIGHTED                (is the cleaning right?)
  3. PDF  vs our data, unweighted              (do the weights matter?)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Topline.pdf'
SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
FINAL = ROOT / 'data/clean/pew_clean_final.csv'
REF = ROOT / 'references/topline_expected.csv'

# ---------- 1. read page 48 of the PDF ----------
text = PdfReader(str(PDF)).pages[47].extract_text()          # page 48 (counting from 1)
q30_part, q31_part = text.split('Q31.')
# the PDF text lists the 24 country names first, then the 24 rows of numbers, in the same order
pdf = []
for q, part in (('ai_heard', q30_part), ('ai_cncexc', q31_part)):
    countries = re.findall(r'^\s*([A-Z][A-Za-z ]+?)\s+Spring, 2025', part, re.M)
    numbers = re.findall(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+100\s*$', part, re.M)
    assert len(countries) == len(numbers) == 24, f'{q}: {len(countries)} names, {len(numbers)} rows'
    for c, vals in zip(countries, numbers):
        for code, val in zip((1, 2, 3, 8), vals):
            pdf.append({'question': q, 'country': c.strip(), 'code': code, 'pdf': int(val)})
pdf = pd.DataFrame(pdf)
print(f'PDF page 48: read {len(pdf)} numbers ({pdf.country.nunique()} countries x 2 questions x 4 answers)')
assert len(pdf) == 192, 'could not read all 192 numbers from the PDF'

# ---------- 2. PDF vs the typed-in CSV ----------
ref = pd.read_csv(REF)
m = pdf.merge(ref, on=['question', 'country', 'code'], how='outer', indicator=True)
bad = m[(m._merge != 'both') | (m.pdf != m.expected)]
print(f'\n1. PDF vs topline_expected.csv: {int((m.pdf == m.expected).sum())}/192 identical'
      + ('' if bad.empty else f'\n   DIFFERENT: {bad.to_dict("records")}'))

# ---------- 3. PDF vs our data ----------
raw, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
raw.columns = raw.columns.str.strip().str.lower()
final = pd.read_csv(FINAL, low_memory=False, encoding='utf-8-sig')
assert raw.id.astype(int).tolist() == final.id.astype(int).tolist()
names = {int(k): v for k, v in meta.variable_value_labels['country'].items()}
cname = raw.country.astype(int).map(names)
rnd = lambda x: int(np.floor(x + 0.5))                         # Pew rounds .5 up

ours = []
for _, r in pdf.iterrows():
    mask = cname == r.country
    x, w = raw.loc[mask, r.question], final.loc[mask, 'weight']   # weights from the FINAL file
    hit = x.isin([8, 9]) if r.code == 8 else (x == r.code)        # DK/Refused = codes 8 and 9
    ours.append({'weighted': 100 * w[hit].sum() / w.sum(), 'unweighted': 100 * hit.mean()})
pdf = pd.concat([pdf, pd.DataFrame(ours)], axis=1)
pdf['w_round'] = pdf.weighted.map(rnd)
pdf['u_round'] = pdf.unweighted.map(rnd)

print(f'2. PDF vs our data, weighted:   {int((pdf.w_round == pdf.pdf).sum())}/192 match')
print(f'3. PDF vs our data, unweighted: {int((pdf.u_round == pdf.pdf).sum())}/192 match')
off = pdf[pdf.w_round != pdf.pdf]
if not off.empty:
    print('   weighted mismatches:\n', off.to_string(index=False))

lab = {('ai_heard', 1): 'Q30 A lot', ('ai_cncexc', 1): 'Q31 More excited'}
print('\nExample rows for the report (PDF / ours weighted / ours unweighted):')
for c in ('UK', 'India', 'Japan'):
    for key, name in lab.items():
        r = pdf[(pdf.country == c) & (pdf.question == key[0]) & (pdf.code == key[1])].iloc[0]
        print(f'  {c:6s} {name:17s} {r.pdf:3d}% | {r.weighted:5.1f}% | {r.unweighted:5.1f}%')