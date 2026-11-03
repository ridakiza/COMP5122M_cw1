"""COMP5122M P1 — Task 23: Q31 uncertainty/nonresponse by age and country.

From the repository root:
    python scripts/validate_task23.py --self-test
    python scripts/validate_task23.py

Critical distinction: Turkey's ~5% (18-29) vs ~51% (65+) is the
UNWEIGHTED rate of *Don't know + Refused*, NOT Don't know alone.
Age groups are the verified Task 16 output. Numerator comes from ORIGINAL
Q31 codes and their SAV value-label meanings, not cleaned NaNs alone.
Denominator is EVERY respondent with known age in that country/age group,
including the original DK and refused answers. Weighted versions are reported
separately. No hypothesis tests or causal inferences are made here.

No source files or production pew_clean.csv are overwritten; outputs contain
aggregate data only. Country/age small-N caveats are explicit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from src.cleaning import do_clean, country_names
except ModuleNotFoundError as e:
    raise SystemExit('Task 23 requires src/cleaning.py. Keep that file in the repository src folder.') from e

SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
OUT = ROOT / 'data/clean'
BANDS = ['18-29', '30-49', '50-64', '65+']
EXPECTED_TURKEY = {
    '18-29': (329, 15, 0),
    '30-49': (464, 39, 5),
    '50-64': (176, 46, 3),
    '65+': (75, 35, 3),
}


def nseries(s: pd.Series, *, require_integral: bool = False) -> pd.Series:
    """Parse numeric source values, including whitespace-only CSV blanks."""
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors='raise')
    else:
        t = s.astype('string').str.strip()
        x = pd.to_numeric(t.mask(t.eq('')), errors='raise')
    assert np.isfinite(x.dropna().to_numpy(dtype=float)).all(), 'Nonfinite numeric values'
    if require_integral:
        assert x.dropna().mod(1).eq(0).all(), 'Noninteger response or identifier'
    return x


def normalized_band(s: pd.Series) -> pd.Series:
    return (s.astype('string').str.strip()
            .str.replace('–', '-', regex=False)
            .str.replace('—', '-', regex=False))


def label_codes(meta: object) -> tuple[int, int]:
    """Infer DK and Refused response codes from original Q31 SAV labels.

    We must not conflate Q31 code 8/9 with Q32's substantive 'Not sure'.
    """
    mapping = getattr(meta, 'variable_value_labels', {}) or {}
    matches = [m for k, m in mapping.items() if str(k).strip().lower() == 'ai_cncexc']
    assert len(matches) == 1, 'Expected Q31 SAV value labels (one unique field)'
    labels = {int(float(k)): str(v).lower().replace('’', "'") for k, v in matches[0].items()}
    dk = [code for code, meaning in labels.items()
          if "don't know" in meaning or 'do not know' in meaning]
    refused = [code for code, meaning in labels.items() if 'refus' in meaning]
    assert len(dk) == len(refused) == 1 and dk[0] != refused[0], (
        f'Q31 must have separately identified DK and Refused labels: {labels}')
    assert len(labels) == 5, f'Unexpected Q31 response set: {labels}'
    assert set(labels) == {1, 2, 3, dk[0], refused[0]}, (
        f'Unexpected Q31 substantive codes: {labels}')
    return dk[0], refused[0]


def compare_by_id(left: pd.DataFrame, right: pd.DataFrame, cols: list[str], name: str) -> None:
    """Order-invariant comparisons; source weight tolerated to 1e-10."""
    for dataset in (left, right):
        assert 'id' in dataset, 'No respondent ID'
        assert nseries(dataset.id, require_integral=True).notna().all(), 'Missing ID'
        assert dataset.id.is_unique, 'Duplicate ID'
    a = left.set_index('id').sort_index()
    b = right.set_index('id').sort_index()
    assert a.index.equals(b.index), f'{name}: mismatched respondent IDs'
    for col in cols:
        assert col in a and col in b, f'{name}: {col} missing'
        if col == 'age_band':
            ok = normalized_band(a[col]).fillna('<missing>').equals(
                 normalized_band(b[col]).fillna('<missing>'))
        elif col == 'age_band_recovered':
            ok = np.array_equal(a[col].to_numpy(dtype=bool), b[col].to_numpy(dtype=bool))
        else:
            av = nseries(a[col]).to_numpy(dtype=float, na_value=np.nan)
            bv = nseries(b[col]).to_numpy(dtype=float, na_value=np.nan)
            ok = bool(np.isclose(av, bv, rtol=0, atol=1e-10, equal_nan=True).all())
        assert ok, f'{name}: respondent-level mismatch for {col}'
    print(f'PASS: {name} — {len(a):,} matched IDs, {len(cols)} fields')


def calculate_age_q31(raw: pd.DataFrame, clean: pd.DataFrame,
                      names: dict[int, str], dk_code: int, refusal_code: int,
                      *, strict: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full 24x4 table, unweighted & weighted DK/refused counts and denominators."""
    assert {'id', 'country', 'weight', 'ai_cncexc'}.issubset(raw.columns)
    assert {'id', 'age_band', 'ai_cncexc', 'weight'}.issubset(clean.columns)
    assert len(raw) == len(clean), 'Production cleaning dropped respondents'
    assert nseries(raw.id).equals(nseries(clean.id)), 'Cleaning changed row order or IDs'
    assert raw.id.notna().all() and raw.id.is_unique
    code_country = nseries(raw.country, require_integral=True)
    assert code_country.notna().all() and code_country.isin(names).all(), 'Unknown country'
    weight = nseries(raw.weight)
    assert weight.notna().all() and weight.gt(0).all(), 'Invalid survey weight'
    raw_q31 = nseries(raw.ai_cncexc, require_integral=True)
    assert raw_q31.notna().all(), 'Q31 unexpectedly structurally blank'
    assert raw_q31.isin([1, 2, 3, dk_code, refusal_code]).all(), 'Unknown Q31 code'
    clean_q31 = nseries(clean.ai_cncexc)
    expected_clean = raw_q31.mask(raw_q31.isin([dk_code, refusal_code]))
    assert np.isclose(clean_q31.to_numpy(float), expected_clean.to_numpy(float),
                      atol=0, rtol=0, equal_nan=True).all(), (
                          'Production Q31 cleaning differs from SAV-labelled DK/refusal recode')
    assert np.isclose(nseries(clean.weight).to_numpy(float), weight.to_numpy(float),
                      atol=0, rtol=0, equal_nan=True).all(), 'Production weights changed'
    band = normalized_band(clean.age_band)
    assert band.dropna().isin(BANDS).all(), 'Invalid final age group'

    frame = pd.DataFrame({
        'id': nseries(raw.id),
        'country_code': code_country.astype(int),
        'age_band': band,
        'weight': weight,
        'dk': raw_q31.eq(dk_code),
        'refused': raw_q31.eq(refusal_code),
        'substantive': raw_q31.isin([1, 2, 3]),
    })
    frame['either'] = frame.dk | frame.refused
    assert (frame.dk.astype(int) + frame.refused.astype(int) +
            frame.substantive.astype(int)).eq(1).all(), 'Q31 response categories overlap'
    records, missing_rows = [], []
    for code, country in sorted(names.items(), key=lambda t: t[1]):
        country_frame = frame.loc[frame.country_code.eq(code)]
        missing_n = int(country_frame.age_band.isna().sum())
        missing_rows.append({
            'country_code': int(code), 'country': country,
            'all_respondents': len(country_frame),
            'known_age_band_n': len(country_frame) - missing_n,
            'missing_age_band_n': missing_n,
        })
        for group in BANDS:
            members = country_frame.loc[country_frame.age_band.eq(group)]
            n = len(members)
            n_dk = int(members.dk.sum())
            n_refused = int(members.refused.sum())
            n_either = n_dk + n_refused
            n_valid = int(members.substantive.sum())
            assert n_valid + n_either == n, 'Q31 counts do not sum to subgroup size'
            den = float(members.weight.sum())
            assert (den > 0) if n else (den == 0), 'Incorrect weighted denominator'
            w_dk = float(members.loc[members.dk, 'weight'].sum())
            w_refused = float(members.loc[members.refused, 'weight'].sum())
            w_nonresponse = w_dk + w_refused
            # Explicit independent check using a Boolean mask, not just sums.
            assert np.isclose(w_nonresponse,
                              members.loc[members.either, 'weight'].sum(),
                              rtol=0, atol=1e-10), 'Weighted DK/refused mismatch'
            rec = {
                'country_code': int(code), 'country': country,
                'age_band': group, 'group_n': n, 'q31_substantive_n': n_valid,
                'q31_dont_know_n': n_dk, 'q31_refused_n': n_refused,
                'q31_combined_nonresponse_n': n_either,
                'q31_substantive_pct_unweighted': 100 * n_valid / n if n else np.nan,
                'q31_dont_know_pct_unweighted': 100 * n_dk / n if n else np.nan,
                'q31_refused_pct_unweighted': 100 * n_refused / n if n else np.nan,
                'q31_combined_nonresponse_pct_unweighted': 100 * n_either / n if n else np.nan,
                'sum_weight_all_in_group': den,
                'sum_weight_dont_know': w_dk,
                'sum_weight_refused': w_refused,
                'q31_dont_know_pct_weighted': 100 * w_dk / den if den else np.nan,
                'q31_refused_pct_weighted': 100 * w_refused / den if den else np.nan,
                'q31_combined_nonresponse_pct_weighted': 100 * w_nonresponse / den if den else np.nan,
                'small_group_n_lt_100': n < 100,
            }
            records.append(rec)
        assert sum(r['group_n'] for r in records[-4:]) + missing_n == len(country_frame)
    result = pd.DataFrame.from_records(records)
    missing = pd.DataFrame.from_records(missing_rows)
    assert int(result.group_n.sum()) + int(missing.missing_age_band_n.sum()) == len(frame)

    if strict:
        assert len(frame) == 28_333 and len(names) == 24 and len(result) == 96
        assert int(result.group_n.sum()) == 28_148
        assert int(missing.missing_age_band_n.sum()) == 185
        turkey = result.loc[result.country.eq('Turkey')].set_index('age_band')
        assert len(turkey) == 4
        for group, (n, dk, refuse) in EXPECTED_TURKEY.items():
            actual = turkey.loc[group]
            assert (int(actual.group_n), int(actual.q31_dont_know_n),
                    int(actual.q31_refused_n)) == (n, dk, refuse), (
                        f'Turkey age {group}: surprising counts vs original CSV: {actual.to_dict()}')
        assert int(missing.loc[missing.country.eq('Turkey'), 'missing_age_band_n'].item()) == 4
        youth = turkey.loc['18-29']
        older = turkey.loc['65+']
        assert round(float(youth.q31_combined_nonresponse_pct_unweighted)) == 5
        assert round(float(older.q31_combined_nonresponse_pct_unweighted)) == 51
        # Don't know ALONE is ~47%, NOT 51%, among oldest respondents.
        assert round(float(older.q31_dont_know_pct_unweighted)) == 47
    return result, missing


def assert_table_agreement(sav: pd.DataFrame, csv: pd.DataFrame) -> None:
    """Count columns exact, percentages tolerant to harmless SAV/CSV float formatting."""
    sa = sav.set_index(['country_code', 'age_band']).sort_index()
    ca = csv.set_index(['country_code', 'age_band']).sort_index()
    assert sa.index.equals(ca.index)
    for col in sa.columns:
        if col == 'country':
            assert sa[col].equals(ca[col]), f'SAV/CSV country label mismatch'
        elif pd.api.types.is_numeric_dtype(sa[col]):
            assert np.isclose(sa[col].to_numpy(dtype=float), ca[col].to_numpy(dtype=float),
                              atol=1e-8, rtol=0, equal_nan=True).all(), (
                                  f'SAV/CSV mismatch in {col}')
        else:
            assert sa[col].equals(ca[col]), f'SAV/CSV mismatch in {col}'


def must_fail(action, label: str) -> None:
    try:
        action()
    except (ValueError, AssertionError):
        print(f'PASS: {label}')
    else:
        raise AssertionError(f'Self-test failed to detect: {label}')


def self_test() -> None:
    from types import SimpleNamespace
    meta = SimpleNamespace(variable_value_labels={'ai_cncexc': {
        1: 'More excited than concerned', 2: 'More concerned than excited',
        3: 'Equally concerned and excited', 8: "Don't know (DO NOT READ)",
        9: 'Refused (DO NOT READ)',
    }})
    dk, refused = label_codes(meta)
    assert (dk, refused) == (8, 9)
    toy = pd.DataFrame({
        'id': [10, 11, 12, 13, 14, 15],
        'country': [19, 19, 19, 19, 19, 19],
        'weight': [1, 1, 2, 1, 3, 2],
        'ai_cncexc': [8, 1, 9, 8, 3, 9],
    })
    clean = toy.copy(deep=True)
    clean['ai_cncexc'] = clean.ai_cncexc.mask(clean.ai_cncexc.isin([dk, refused]))
    clean['age_band'] = ['18–29', '18–29', '18–29', '65+', '65+', '65+']
    original = toy.copy(deep=True)
    cells, missing = calculate_age_q31(toy, clean, {19: 'Turkey'}, dk, refused, strict=False)
    youth = cells.loc[cells.age_band.eq('18-29')].iloc[0]
    older = cells.loc[cells.age_band.eq('65+')].iloc[0]
    assert (youth.group_n, youth.q31_dont_know_n, youth.q31_refused_n) == (3, 1, 1)
    assert np.isclose(youth.q31_combined_nonresponse_pct_unweighted, 200 / 3)
    assert np.isclose(youth.q31_combined_nonresponse_pct_weighted, 75.0)
    assert (older.group_n, older.q31_dont_know_n, older.q31_refused_n) == (3, 1, 1)
    assert np.isclose(older.q31_combined_nonresponse_pct_weighted, 50.0)
    pd.testing.assert_frame_equal(toy, original, check_exact=True)
    print('PASS: unweighted and weighted DK/Refused calculations, all-group denominator and raw unchanged')
    ref_missing = cells.copy(deep=True)
    assert int(missing.missing_age_band_n.sum()) == 0
    altered = clean.copy(deep=True)
    altered.loc[altered.id.eq(10), 'ai_cncexc'] = 8
    must_fail(lambda: calculate_age_q31(toy, altered, {19: 'Turkey'}, dk, refused, strict=False),
              'Q31 code 8 left as an analytical answer detected')
    altered = toy.copy(deep=True)
    altered.loc[altered.id.eq(10), 'ai_cncexc'] = 10
    must_fail(lambda: calculate_age_q31(altered, clean, {19: 'Turkey'}, dk, refused, strict=False),
              'unexpected source Q31 code detected')
    altered = toy.copy(deep=True)
    altered.loc[altered.id.eq(10), 'id'] = 11
    must_fail(lambda: calculate_age_q31(altered, clean, {19: 'Turkey'}, dk, refused, strict=False),
              'duplicate respondent ID detected')
    altered = toy.copy(deep=True)
    altered.loc[altered.id.eq(10), 'weight'] = 4.0
    must_fail(lambda: compare_by_id(toy, altered, ['weight'], 'synthetic mutation'),
              'SAV/CSV respondent weight discrepancy detected')
    wrong_meta = SimpleNamespace(variable_value_labels={'ai_cncexc': {
        1: 'Excited', 2: 'Concerned', 3: 'Equal', 8: 'Other opinion', 9: 'Refused'}})
    must_fail(lambda: label_codes(wrong_meta), 'missing DK meaning in SAV metadata detected')
    altered = clean.copy(deep=True)
    altered.loc[altered.id.eq(10), 'age_band'] = '30–49'
    must_fail(lambda: compare_by_id(clean, altered, ['age_band'], 'synthetic age mutation'),
              'respondent-level age-band discrepancy detected')
    print('TASK 23 SELF-TEST PASSED')


def main() -> None:
    try:
        import pyreadstat
    except ImportError as e:
        raise SystemExit('Install requirements in your venv (including pyreadstat)') from e
    assert SAV.is_file(), f'Original SAV missing: {SAV}'
    assert CSV.is_file(), f'Original CSV missing: {CSV}'
    print('=== TASK 23: Q31 DK / REFUSAL RATES BY AGE AND COUNTRY ===')
    raw_sav, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    raw_csv = pd.read_csv(CSV, encoding='utf-8-sig', low_memory=False)
    raw_sav.columns = raw_sav.columns.str.strip().str.lower()
    raw_csv.columns = raw_csv.columns.str.strip().str.lower()
    assert raw_sav.shape == raw_csv.shape == (28_333, 483), 'Unexpected source size'
    originals = (raw_sav.copy(deep=True), raw_csv.copy(deep=True))
    dk, refused = label_codes(meta)
    print(f'Original SAV Q31 labels: DK code={dk}; Refused code={refused}')
    names = country_names(meta)
    assert len(names) == 24
    compare_by_id(raw_sav, raw_csv,
                  ['country', 'weight', 'age', 'age_australia', 'ai_cncexc'],
                  'original SAV/CSV (country, weight, age and Q31)')
    prod_sav = do_clean(raw_sav, meta, log=[])
    prod_csv = do_clean(raw_csv, meta, log=[])
    pd.testing.assert_frame_equal(originals[0], raw_sav, check_exact=True)
    pd.testing.assert_frame_equal(originals[1], raw_csv, check_exact=True)
    print('PASS: production cleaning left both original input frames unchanged')
    compare_by_id(prod_sav, prod_csv,
                  ['country', 'weight', 'age_band', 'age_band_recovered', 'ai_cncexc'],
                  'cleaned SAV/CSV Q31 + age groups')
    sav_table, sav_missing = calculate_age_q31(raw_sav, prod_sav, names, dk, refused)
    csv_table, csv_missing = calculate_age_q31(raw_csv, prod_csv, names, dk, refused)
    assert_table_agreement(sav_table, csv_table)
    pd.testing.assert_frame_equal(sav_missing, csv_missing, check_exact=True)
    print('PASS: all 96/96 country × age cells match SAV/CSV (counts, unweighted and weighted)')
    print('PASS: 28,148 known ages, 185 unknown ages, all 28,333 people preserved')
    turkey = sav_table.loc[sav_table.country.eq('Turkey')].copy()
    turkey['age_band'] = pd.Categorical(turkey.age_band, categories=BANDS, ordered=True)
    turkey = turkey.sort_values('age_band')
    print('\nTURKEY Q31 — original labels and TOTAL AGE-GROUP denominator:')
    print(turkey[[
        'age_band', 'group_n', 'q31_dont_know_n', 'q31_refused_n',
        'q31_combined_nonresponse_n', 'q31_dont_know_pct_unweighted',
        'q31_combined_nonresponse_pct_unweighted',
        'q31_combined_nonresponse_pct_weighted',
    ]].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    youngest = turkey.loc[turkey.age_band.eq('18-29')].iloc[0]
    eldest = turkey.loc[turkey.age_band.eq('65+')].iloc[0]
    gap = (eldest.q31_combined_nonresponse_pct_unweighted
           - youngest.q31_combined_nonresponse_pct_unweighted)
    print(f'\nTurkey combined DK+Refused (UNWEIGHTED): '
          f'18–29 = {youngest.q31_combined_nonresponse_pct_unweighted:.4f}% '
          f'({youngest.q31_combined_nonresponse_n}/{youngest.group_n}) -> 5%; '
          f'65+ = {eldest.q31_combined_nonresponse_pct_unweighted:.4f}% '
          f'({eldest.q31_combined_nonresponse_n}/{eldest.group_n}) -> 51%.')
    print(f'Unweighted combined nonresponse gap (oldest minus youngest): {gap:.4f} percentage points')
    print(f'CAUTION: Turkey 65+ DK ALONE = {eldest.q31_dont_know_pct_unweighted:.4f}% '
          f'(35/75), NOT 51%. Three refusals make combined nonresponse 38/75.')
    print(f'Weighted combined nonresponse: youngest '
          f'{youngest.q31_combined_nonresponse_pct_weighted:.4f}%, '
          f'oldest {eldest.q31_combined_nonresponse_pct_weighted:.4f}%.')
    print('CAUTION: only 75 respondents in Turkey 65+; descriptive comparison only, '
          'not a causal effect or a design-corrected significance test.')
    print('All 24 countries included; 4 Turkey respondents and 185 overall '
          'have unknown age bands, so are not assigned to age-specific denominators.')

    OUT.mkdir(parents=True, exist_ok=True)
    sav_table.to_csv(OUT / 'task23_country_age_q31_nonresponse_audit.csv', index=False)
    turkey.to_csv(OUT / 'task23_turkey_age_q31_detail.csv', index=False)
    sav_missing.to_csv(OUT / 'task23_age_missing_audit.csv', index=False)
    pd.DataFrame([{
        'task': 23,
        'decision': 'Report Q31 age-specific nonresponse with DK and Refused separately and combined; denominator includes everyone with known age in each country × age group',
        'alternative': 'Report the ~51% figure as DK alone or use only valid Q31 answers as denominator; ignore survey weights or equate weighted and unweighted results',
        'reason': 'Turkey 65+ has 35 DK and 3 Refused out of 75; 51% means 38/75 combined, whereas DK alone is 35/75=46.67%',
        'effect': 'Turkey 18-29 15/329=4.56% vs Turkey 65+ 38/75=50.67% combined unweighted; describe weighted values separately',
        'statistical_caveat': 'Descriptive only; Turkey 65+ N=75 and unequal weights/full survey design affect precision',
        'verification': 'Original Q31 SAV labels; SAV/CSV respondent-level input and output; all 96 country-age weighted and unweighted cells compared',
        'status': 'VERIFIED_BY_STUDENT_SAV_RUN',
    }]).to_csv(OUT / 'task23_decision_log_entry.csv', index=False)
    pd.DataFrame([{
        'task': 23,
        'ai_contribution': 'Developed Q31 age-by-country audit and synthetic error tests; separated DK from Refused and weighted from unweighted rates',
        'independent_verification': 'Original CSV Turkey codes 15/329 (18-29) and 35+3/75 (65+); original SAV labels verified at runtime',
        'human_verification': 'Student execution against original SAV plus CSV; logs printed and saved',
        'ai_discrepancy': 'Earlier unqualified phrase 5% vs 51% "Don\'t know" incorrectly implies DK alone; in the oldest group 51% is combined DK+Refused',
        'correction': 'State explicitly 35/75=46.67% DK alone and 38/75=50.67% DK+Refused; preserve raw codes and include refusals in the denominator',
    }]).to_csv(OUT / 'task23_ai_use_log_entry.csv', index=False)
    pd.DataFrame([{
        'task': 23,
        'issue': 'Unqualified description of Turkey 5% versus 51% as "Don\'t know"',
        'original_claim': 'Turkey age 18-29 ~5% versus age 65+ ~51% "Don\'t know"',
        'how_detected': 'Separated original Q31 DK (8) from Refused (9) using SAV labels; recomputed age-specific response counts',
        'corrected_finding': 'Turkey age 65+: DK only 35/75=46.67%, Refused 3/75=4.00%, combined 38/75=50.67%; age 18-29 combined 15/329=4.56%',
        'effect_on_original_data': 'None: terminology/interpretation correction only',
        'status': 'CORRECTED',
    }]).to_csv(OUT / 'task23_ai_discrepancy_entry.csv', index=False)
    (OUT / 'task23_validation_note.txt').write_text(
        'Task 23 — Turkey Q31 (AI more concerned vs excited) age nonresponse.\n'
        'Original Q31 SAV labels identify DK and Refused, which are reported '
        'separately and jointly. Unweighted denominator is ALL respondents '
        'with a known age band in the country/age cell (including nonresponses); '
        'weighted denominator is the sum of all respondents\' survey weights '
        'in that cell. The cleaned Q31 analytical column uses NaN for DK/refused, '
        'but the original numeric labels are necessary to separate them.\n'
        'Turkey ages 18-29: DK=15, Refused=0, N=329 => combined 4.5593%.\n'
        'Turkey ages 65+: DK=35, Refused=3, N=75 => combined 50.6667%; '
        'DK alone =46.6667%. Unweighted combined gap 46.1074 percentage points.\n'
        'All 96 country-by-age cells verified, with weighted/nonweighted rates '
        'and 185 missing age bands kept unclassified. Small Ns and complex survey '
        'design prevent naive precision or significance claims.\n'
        'No source data or shared pew_clean.csv overwritten.\n', encoding='utf-8')
    print('\nSaved: task23_country_age_q31_nonresponse_audit.csv, '
          'task23_turkey_age_q31_detail.csv, task23_age_missing_audit.csv,')
    print('       task23_decision_log_entry.csv, task23_ai_use_log_entry.csv,')
    print('       task23_ai_discrepancy_entry.csv, task23_validation_note.txt')
    print('TASK 23 PASSED: Turkey 5% vs 51% is COMBINED DK+Refused unweighted; '
          'DK alone 65+=47%; all 96 age-country cells match SAV/CSV; no rows changed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
