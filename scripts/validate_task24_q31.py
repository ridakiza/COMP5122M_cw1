"""COMP5122M P1 - Task 24 (part 2): the same Don't-know test for Q31.

Run from the repository root (after validate_task24.py, which covers Q30):
    python scripts/validate_task24_q31.py --self-test
    python scripts/validate_task24_q31.py

Why: Task 23 showed older people say Don't know on Q31 far more than young
people (Turkey 6% vs 59% weighted). Task 24 part 1 only tested Q30. This applies
the same test to Q31: weighted % for each answer, 18-29 minus 65+, with
Don't know/Refused KEPT in the base vs LEFT OUT of the base.

Codes come from the .sav labels. Age groups come from src/cleaning.py (checked
in Tasks 18 and 21). Nothing is changed or saved except aggregate audit files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cleaning import do_clean, country_names  # noqa: E402

SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
OUT = ROOT / 'data/clean'
YOUNG, OLD = '18-29', '65+'


def q31_codes(meta) -> tuple[dict[int, str], set[int]]:
    """Answer labels and the no-answer codes, read from the .sav labels."""
    labels = {int(float(k)): str(v) for k, v in meta.variable_value_labels['ai_cncexc'].items()}
    norm = {k: v.lower().replace('’', "'") for k, v in labels.items()}
    missing = {k for k, v in norm.items() if "don't know" in v or 'refused' in v}
    answers = {k: labels[k] for k in labels if k not in missing}
    assert set(answers) == {1, 2, 3} and len(missing) == 2, f'Unexpected Q31 labels: {labels}'
    return answers, missing


def gaps(df: pd.DataFrame, answers: dict[int, str], missing: set[int]) -> pd.DataFrame:
    """One row per country x answer: weighted young-minus-old gap, DK kept vs left out."""
    rows = []
    for country, g in df.groupby('country', sort=True):
        for code, label in answers.items():
            out = {'country': country, 'answer_code': code, 'answer': label}
            for keep in (True, False):
                pct = {}
                for band in (YOUNG, OLD):
                    s = g[g.age_band == band]
                    base = s if keep else s[~s.q31.isin(missing)]
                    assert base.weight.sum() > 0, f'{country} {band}: empty base'
                    pct[band] = 100 * base.weight[base.q31 == code].sum() / base.weight.sum()
                tag = 'dk_kept' if keep else 'dk_left_out'
                out[f'young_pct_{tag}'] = pct[YOUNG]
                out[f'old_pct_{tag}'] = pct[OLD]
                out[f'gap_{tag}'] = pct[YOUNG] - pct[OLD]
            out['change_pp'] = out['gap_dk_left_out'] - out['gap_dk_kept']
            rows.append(out)
    return pd.DataFrame(rows)


def self_test() -> None:
    # Young: 2 concerned, 0 DK. Old: 1 concerned, 1 DK (all weights 1).
    toy = pd.DataFrame({'country': ['X'] * 4, 'age_band': [YOUNG, YOUNG, OLD, OLD],
                        'weight': [1.0] * 4, 'q31': [2, 2, 2, 8]})
    t = gaps(toy, {1: 'excited', 2: 'concerned', 3: 'equally'}, {8, 9})
    r = t[t.answer_code == 2].iloc[0]
    assert np.isclose(r.gap_dk_kept, 100 - 50) and np.isclose(r.gap_dk_left_out, 100 - 100)
    assert np.isclose(r.change_pp, -50)
    print('PASS: DK kept in base gives a 50-point gap; DK left out gives 0 (hand-checked)')
    print('TASK 24 Q31 SELF-TEST PASSED')


def main() -> None:
    import pyreadstat
    print("=== TASK 24 (Q31): GENERATION GAP WITH DON'T KNOW KEPT vs LEFT OUT ===")
    raw, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    raw.columns = raw.columns.str.strip().str.lower()
    answers, missing = q31_codes(meta)
    print(f'Q31 answers from .sav labels: {answers}; no-answer codes: {sorted(missing)}')
    before = raw.copy(deep=True)
    clean = do_clean(raw, meta, log=[])
    pd.testing.assert_frame_equal(raw, before, check_exact=True)
    assert len(clean) == 28_333 and clean.id.equals(raw.id)
    df = pd.DataFrame({
        'country': raw.country.astype(int).map(country_names(meta)),
        'age_band': clean.age_band.astype('string').str.replace('–', '-', regex=False),
        'weight': raw.weight.astype(float),
        'q31': raw.ai_cncexc.astype(int),   # raw codes: 8/9 still present
    })
    assert int(df.q31.isin(missing).sum()) == 1964, 'Q31 no-answer count changed (Task 10: 1,964)'
    t = gaps(df, answers, missing)
    assert len(t) == 24 * 3

    print('\nMean absolute change in the young-minus-old gap across 24 countries (points):')
    summary = (t.groupby(['answer_code', 'answer']).change_pp
               .agg(mean_abs=lambda s: s.abs().mean(), largest_abs=lambda s: s.abs().max())
               .reset_index())
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    conc = t[t.answer_code == 2].reindex(
        t[t.answer_code == 2].change_pp.abs().sort_values(ascending=False).index)
    print('\n"More concerned" gap (18-29 minus 65+), biggest changes:')
    print(conc[['country', 'gap_dk_kept', 'gap_dk_left_out', 'change_pp']].head(8)
          .to_string(index=False, float_format=lambda x: f'{x:.1f}'))

    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / 'task24_q31_country_audit.csv', index=False)
    summary.to_csv(OUT / 'task24_q31_summary.csv', index=False)
    print('\nSaved: task24_q31_country_audit.csv, task24_q31_summary.csv')
    print("TASK 24 (Q31) PASSED: 24 countries x 3 answers; 28,333 people kept; no data changed.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    self_test() if a.self_test else main()