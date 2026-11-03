"""COMP5122M P1 — Task 24: sensitivity to Q30 nonresponse denominators.

Run from repository root:
    python scripts/validate_task24.py --self-test
    python scripts/validate_task24.py

Optional PRELIMINARY CSV diagnostic when SAV unavailable:
    python scripts/validate_task24.py --csv-diagnostic
This diagnostic reads the existing clean CSV snapshot, rather than executing
SAV-metadata-backed production cleaning. It is NOT a replacement for SAV check.

Demonstration measure is the WEIGHTED Q30 'A lot' awareness gap:
    percentage ages 18–29 MINUS percentage ages 65+ (per country).
Baseline includes Q30 DK+Refused in denominators, as Pew's official topline
percentages do. Sensitivity EXCLUDES BOTH DK AND REFUSED from denominators.
Numerators and weight sums are built afresh from original response codes.
This is a denominator sensitivity, not a change to the dataset or a significance
test. Other sensitivity dimensions belong to Tasks 25–27.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
try:
    from src.cleaning import do_clean, country_names, awareness_gap
    from validate_task21 import check_independent, assert_same_by_id
except ModuleNotFoundError as exc:
    raise SystemExit(
        'Task 24 requires src/cleaning.py and scripts/validate_task21.py. '
        f'Missing dependency: {exc}'
    ) from exc

SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
CLEAN_SNAPSHOT = ROOT / 'data/clean/pew_clean.csv'
OUT = ROOT / 'data/clean'
BANDS = ('18-29', '65+')
EXPECTED_N = 28_333
EXPECTED_COUNTRIES = 24
EXPECTED_ABS_MEAN_CHANGE = 0.5057911172399615  # check only; NOT used in calculation


def numeric(s: pd.Series) -> pd.Series:
    """Strict numeric coercion; blank/whitespace is NA, text is an error."""
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors='raise')
    else:
        text = s.astype('string').str.strip()
        v = pd.to_numeric(text.mask(text.eq('')), errors='raise')
    assert np.isfinite(v.dropna().to_numpy(dtype=float)).all(), 'Nonfinite numeric response'
    return v


def band_norm(s: pd.Series) -> pd.Series:
    return (s.astype('string').str.strip()
            .str.replace('–', '-', regex=False)
            .str.replace('—', '-', regex=False))


def q30_labels(meta: object) -> tuple[int, int, int]:
    """Identify A lot / DK / refused FROM SAV response labels, never by numeric guess."""
    all_labels = getattr(meta, 'variable_value_labels', {}) or {}
    matches = [v for k, v in all_labels.items() if str(k).strip().lower() == 'ai_heard']
    assert len(matches) == 1, 'Missing or ambiguous original SAV labels for ai_heard'
    labels: dict[int, str] = {}
    for code, text in matches[0].items():
        f = float(code)
        assert f.is_integer(), 'Nonintegral Q30 response code'
        labels[int(f)] = str(text).strip().lower().replace('’', "'")
    positive = [code for code, lab in labels.items() if lab == 'a lot']
    dk = [code for code, lab in labels.items() if "don't know" in lab or 'do not know' in lab]
    refusal = [code for code, lab in labels.items() if 'refus' in lab]
    assert len(positive) == len(dk) == len(refusal) == 1, (
        f'Cannot identify Q30 A lot, DK and Refused from SAV labels: {labels}')
    assert len(labels) == 5 and set(labels) == {1, 2, 3, dk[0], refusal[0]}, (
        f'Unexpected Q30 codebook definitions: {labels}')
    assert len(set((positive[0], dk[0], refusal[0]))) == 3
    return positive[0], dk[0], refusal[0]


def calculate(raw: pd.DataFrame, cleaned: pd.DataFrame,
              names: dict[int, str], positive: int, dk: int, refused: int,
              *, strict: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Independent per-country estimates from ORIGINAL Q30 response codes.

    Age groups are supplied by Task 16's production cleaning, and are checked
    against an independently developed classifier during the full SAV run.
    """
    assert {'id', 'country', 'weight', 'ai_heard'}.issubset(raw.columns)
    assert {'id', 'country', 'weight', 'ai_heard', 'age_band'}.issubset(cleaned.columns)
    assert len(raw) == len(cleaned), 'Cleaning removed respondents'
    assert raw.id.notna().all() and raw.id.is_unique, 'Original IDs invalid'
    assert cleaned.id.notna().all() and cleaned.id.is_unique, 'Cleaned IDs invalid'
    assert np.array_equal(numeric(raw.id).to_numpy(dtype=np.int64),
                          numeric(cleaned.id).to_numpy(dtype=np.int64)), 'IDs/order changed'

    code_country = numeric(raw.country)
    assert code_country.notna().all() and code_country.mod(1).eq(0).all()
    assert code_country.isin(names).all(), 'Missing country name in metadata'
    clean_country = numeric(cleaned.country)
    assert np.array_equal(code_country.to_numpy(dtype=float),
                          clean_country.to_numpy(dtype=float)), 'Country changed by cleaning'
    wt = numeric(raw.weight)
    cwt = numeric(cleaned.weight)
    assert wt.notna().all() and wt.gt(0).all(), 'Invalid survey weight'
    assert np.isclose(wt.to_numpy(float), cwt.to_numpy(float), rtol=0, atol=1e-12).all(), (
        'Production weights changed')

    q30 = numeric(raw.ai_heard)
    assert q30.notna().all() and q30.mod(1).eq(0).all(), 'Unexpected blank/noninteger Q30'
    assert q30.isin([1, 2, 3, dk, refused]).all(), 'Unknown Q30 response code'
    clean_q30 = numeric(cleaned.ai_heard)
    expected_clean = q30.mask(q30.isin([dk, refused]))
    assert np.isclose(clean_q30.to_numpy(float), expected_clean.to_numpy(float),
                      rtol=0, atol=0, equal_nan=True).all(), (
        'Production Q30 recoding is not identical to SAV-labelled DK/refusal')
    age_band = band_norm(cleaned.age_band)
    assert age_band.dropna().isin(('18-29', '30-49', '50-64', '65+')).all(), (
        'Unrecognised age group')

    frame = pd.DataFrame({
        'id': numeric(raw.id).to_numpy(dtype=np.int64),
        'country': code_country.astype(int).map(names).to_numpy(),
        'age_band': age_band.to_numpy(),
        'weight': wt.to_numpy(float),
        'a_lot': q30.eq(positive).to_numpy(bool),
        'nonresponse': q30.isin((dk, refused)).to_numpy(bool),
    })
    assert not (frame.a_lot & frame.nonresponse).any()
    country_rows: list[dict] = []
    group_rows: list[dict] = []
    for country in sorted(names.values()):
        country_frame = frame.loc[frame.country.eq(country)]
        result = {}
        for group in BANDS:
            sub = country_frame.loc[country_frame.age_band.eq(group)]
            n_all, n_nonresponse = len(sub), int(sub.nonresponse.sum())
            n_valid = n_all - n_nonresponse
            assert n_all > 0 and n_valid > 0, f'{country} {group}: empty denominator'
            n_positive = int(sub.a_lot.sum())
            sum_w_all = float(sub.weight.sum())
            sum_w_valid = float(sub.loc[~sub.nonresponse, 'weight'].sum())
            sum_w_positive = float(sub.loc[sub.a_lot, 'weight'].sum())
            assert 0 <= sum_w_positive <= sum_w_valid <= sum_w_all
            pct_all = 100.0 * sum_w_positive / sum_w_all
            pct_valid = 100.0 * sum_w_positive / sum_w_valid
            assert pct_valid + 1e-12 >= pct_all >= 0
            group_rows.append({
                'country': country, 'age_band': group, 'all_age_group_n': n_all,
                'nonresponse_dk_refused_n': n_nonresponse,
                'substantive_valid_n': n_valid, 'a_lot_n': n_positive,
                'sum_weight_all': sum_w_all, 'sum_weight_valid': sum_w_valid,
                'sum_weight_a_lot': sum_w_positive,
                'a_lot_pct_baseline_dk_in_denominator': pct_all,
                'a_lot_pct_scenario_exclude_dk_refused': pct_valid,
            })
            result[group] = (pct_all, pct_valid)
        gap_base = result['18-29'][0] - result['65+'][0]
        gap_sens = result['18-29'][1] - result['65+'][1]
        country_rows.append({
            'country': country, 'baseline_gap_pp': gap_base,
            'exclude_dk_refused_gap_pp': gap_sens,
            'change_pp': gap_sens - gap_base,
            'abs_change_pp': abs(gap_sens - gap_base),
        })
    countries = pd.DataFrame(country_rows)
    groups = pd.DataFrame(group_rows)
    assert len(groups) == len(names) * 2
    assert len(countries) == len(names)

    if strict:
        assert len(raw) == EXPECTED_N and len(names) == EXPECTED_COUNTRIES
        assert int(age_band.notna().sum()) == 28_148
        assert int(age_band.isna().sum()) == 185
        assert int(q30.isin((dk, refused)).sum()) == 808
        assert int(q30.eq(positive).sum()) == 9407
        actual = float(countries.abs_change_pp.mean())
        assert np.isclose(actual, EXPECTED_ABS_MEAN_CHANGE, rtol=0, atol=1e-6), (
            f'Unexpected average absolute country gap change: {actual:.9f} pp')
    return countries, groups


def assert_country_agreement(a: pd.DataFrame, b: pd.DataFrame,
                             context: str) -> None:
    """All 24 country results match, to numerical precision."""
    x = a.sort_values('country').reset_index(drop=True)
    y = b.sort_values('country').reset_index(drop=True)
    assert x.country.equals(y.country), f'{context}: countries differ'
    for col in (c for c in x.columns if c != 'country'):
        assert np.isclose(x[col].to_numpy(float), y[col].to_numpy(float),
                          rtol=0, atol=1e-9).all(), f'{context}: mismatch in {col}'


def assert_production_agreement(clean: pd.DataFrame, independent: pd.DataFrame) -> None:
    """Compare independently calculated gaps with src.cleaning.awareness_gap()."""
    base = awareness_gap(clean, keep_dk=True, weights=True)
    alt = awareness_gap(clean, keep_dk=False, weights=True)
    expected = independent.set_index('country')
    for mode, v in [('baseline', base), ('exclude_dk_refused', alt)]:
        check = v.set_index('country')['gap_pp'].reindex(expected.index)
        assert check.notna().all(), f'{mode}: missing production country'
        column = 'baseline_gap_pp' if mode == 'baseline' else 'exclude_dk_refused_gap_pp'
        assert np.isclose(check.to_numpy(float), expected[column].to_numpy(float),
                          rtol=0, atol=1e-9).all(), f'{mode}: production gap differs'


def fail_on_assertion(fn, label: str) -> None:
    try:
        fn()
    except AssertionError:
        print('PASS:', label)
    else:
        raise AssertionError(f'Self-test did not detect error: {label}')


def self_test() -> None:
    fake_labels = SimpleNamespace(variable_value_labels={
        'ai_heard': {1: 'A lot', 2: 'A little', 3: 'Nothing at all',
                     8: "Don't know (DO NOT READ)", 9: 'Refused (DO NOT READ)'},
    })
    pos, dk, refused = q30_labels(fake_labels)
    assert (pos, dk, refused) == (1, 8, 9)
    raw = pd.DataFrame({
        'id': [10, 11, 12, 13, 14, 15],
        'country': [5] * 6,
        'weight': [2., 1., 1., 1., 1., 2.],
        'ai_heard': [1, 8, 2, 1, 9, 2],
    })
    clean = raw.copy(deep=True)
    clean['ai_heard'] = raw.ai_heard.mask(raw.ai_heard.isin((dk, refused)))
    clean['age_band'] = ['18–29'] * 3 + ['65+'] * 3
    original = raw.copy(deep=True)
    table, groups = calculate(raw, clean, {5: 'Toy Country'}, pos, dk, refused,
                              strict=False)
    youth = groups.loc[groups.age_band.eq('18-29')].iloc[0]
    older = groups.loc[groups.age_band.eq('65+')].iloc[0]
    assert (youth.all_age_group_n, youth.nonresponse_dk_refused_n,
            youth.substantive_valid_n, youth.a_lot_n) == (3, 1, 2, 1)
    assert (older.all_age_group_n, older.nonresponse_dk_refused_n,
            older.substantive_valid_n, older.a_lot_n) == (3, 1, 2, 1)
    assert np.isclose(youth.a_lot_pct_baseline_dk_in_denominator, 50.0)
    assert np.isclose(youth.a_lot_pct_scenario_exclude_dk_refused, 200 / 3)
    assert np.isclose(older.a_lot_pct_baseline_dk_in_denominator, 25.0)
    assert np.isclose(older.a_lot_pct_scenario_exclude_dk_refused, 100 / 3)
    assert np.isclose(table.baseline_gap_pp.iloc[0], 25)
    assert np.isclose(table.exclude_dk_refused_gap_pp.iloc[0], 100 / 3)
    assert np.isclose(table.change_pp.iloc[0], 25 / 3)
    pd.testing.assert_frame_equal(raw, original, check_exact=True)
    print('PASS: weighted DK-inclusive vs DK/refusal-excluded denominator; synthetic gap correct')
    bad = raw.copy(deep=True)
    bad.loc[0, 'ai_heard'] = 7
    fail_on_assertion(lambda: calculate(bad, clean, {5: 'Toy Country'}, pos, dk, refused,
                                        strict=False), 'unknown original response code rejected')
    bad = clean.copy(deep=True)
    bad.loc[bad.id.eq(11), 'ai_heard'] = 8
    fail_on_assertion(lambda: calculate(raw, bad, {5: 'Toy Country'}, pos, dk, refused,
                                        strict=False), 'failure to recode DK detected')
    bad = clean.copy(deep=True)
    bad.loc[bad.id.eq(11), 'age_band'] = '65+'
    fail_on_assertion(lambda: assert_country_agreement(table,
        calculate(raw, bad, {5: 'Toy Country'}, pos, dk, refused, strict=False)[0],
        'self-test age misclassification'), 'different age-band allocation changes gap')
    bad = raw.copy(deep=True)
    bad.loc[0, 'weight'] = 0
    fail_on_assertion(lambda: calculate(bad, clean, {5: 'Toy Country'}, pos, dk, refused,
                                        strict=False), 'nonpositive weight rejected')
    wrong_meta = SimpleNamespace(variable_value_labels={
        'ai_heard': {1: 'A lot', 2: 'A little', 3: 'Nothing at all',
                     8: 'Not sure', 9: 'Refused'},
    })
    fail_on_assertion(lambda: q30_labels(wrong_meta), 'unknown DK label not silently guessed')
    print('TASK 24 SELF-TEST PASSED')


def diagnostic() -> None:
    """CSV-only diagnostic using an EXISTING cleaned snapshot, not original SAV."""
    assert CSV.is_file() and CLEAN_SNAPSHOT.is_file(), (
        'CSV diagnostic requires original CSV and existing data/clean/pew_clean.csv')
    print('=== TASK 24: CSV DIAGNOSTIC ONLY (not SAV verification) ===')
    raw = pd.read_csv(CSV, low_memory=False)
    clean = pd.read_csv(CLEAN_SNAPSHOT, low_memory=False)
    assert raw.shape == (EXPECTED_N, 483) and len(clean) == EXPECTED_N
    names = {}
    for c, name in clean[['country', 'country_name']].drop_duplicates().itertuples(index=False):
        names[int(c)] = str(name)
    assert len(names) == EXPECTED_COUNTRIES
    assert_same_by_id(raw, clean, ['country', 'weight'], context='CSV source vs cleaned snapshot')
    results, ages = calculate(raw, clean, names, 1, 8, 9)
    assert_production_agreement(clean, results)
    print(f'PASS: 24 countries, mean absolute change = {results.abs_change_pp.mean():.6f} pp')
    print('NOTE: CSV diagnostic uses previously recorded Q30 code meanings and saved clean snapshot;')
    print('      actual SAV verification must read ORIGINAL SAV value labels at runtime.')


def main() -> None:
    try:
        import pyreadstat
    except ImportError as e:
        raise SystemExit('Install pyreadstat in the active venv before original SAV validation') from e
    assert SAV.is_file() and CSV.is_file(), 'Missing original SAV or CSV from data/raw/'
    print('=== TASK 24: Q30 DENOMINATOR SENSITIVITY — EXCLUDE DK + REFUSED ===')
    raw_sav, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False,
                                       user_missing=True)
    raw_csv = pd.read_csv(CSV, encoding='utf-8-sig', low_memory=False)
    raw_sav.columns = raw_sav.columns.str.strip().str.lower()
    raw_csv.columns = raw_csv.columns.str.strip().str.lower()
    assert raw_sav.shape == raw_csv.shape == (EXPECTED_N, 483), 'Unexpected source shape'
    a_lot, dk, refused = q30_labels(meta)
    print(f'SAV Q30 labels: A lot={a_lot}; Don\'t know={dk}; Refused={refused}')
    names = country_names(meta)
    assert_same_by_id(raw_sav, raw_csv,
                      ['country', 'weight', 'ai_heard', 'age', 'age_australia'],
                      context='Original SAV vs CSV')
    before_sav = raw_sav.copy(deep=True)
    before_csv = raw_csv.copy(deep=True)
    prod_sav = do_clean(raw_sav, meta, log=[])
    prod_csv = do_clean(raw_csv, meta, log=[])
    pd.testing.assert_frame_equal(raw_sav, before_sav, check_exact=True)
    pd.testing.assert_frame_equal(raw_csv, before_csv, check_exact=True)
    print('PASS: original SAV and CSV input frames unchanged')
    assert_same_by_id(prod_sav, prod_csv,
                      ['country', 'weight', 'ai_heard', 'age_band', 'age_band_recovered'],
                      context='Cleaned SAV vs CSV')
    check_independent(raw_sav, prod_sav, meta, name='SAV')
    check_independent(raw_csv, prod_csv, meta, name='CSV')
    sav_audit, sav_groups = calculate(raw_sav, prod_sav, names, a_lot, dk, refused)
    csv_audit, csv_groups = calculate(raw_csv, prod_csv, names, a_lot, dk, refused)
    assert_country_agreement(sav_audit, csv_audit, 'SAV/CSV gap calculations')
    assert np.array_equal(sav_groups[['country','age_band','all_age_group_n',
                                      'nonresponse_dk_refused_n','substantive_valid_n','a_lot_n']].to_numpy(),
                          csv_groups[['country','age_band','all_age_group_n',
                                      'nonresponse_dk_refused_n','substantive_valid_n','a_lot_n']].to_numpy())
    for col in (c for c in sav_groups if c.startswith('sum_weight') or c.startswith('a_lot_pct')):
        assert np.isclose(sav_groups[col].to_numpy(float),
                          csv_groups[col].to_numpy(float), atol=1e-9, rtol=0).all(), (
            f'Country/age SAV/CSV numerator/denominator mismatch: {col}')
    assert_production_agreement(prod_sav, sav_audit)
    assert_production_agreement(prod_csv, csv_audit)
    print('PASS: SAV/CSV all 24 country gaps and 48 group denominators match')
    print('PASS: independent raw-Q30 calculations match production awareness_gap()')

    mean_abs = float(sav_audit.abs_change_pp.mean())
    mean_signed = float(sav_audit.change_pp.mean())
    largest = sav_audit.loc[sav_audit.abs_change_pp.idxmax()]
    print('\nSensitivity: weighted Q30 A lot (18–29 minus 65+) in each country')
    print('Baseline denominator: ALL respondents, including DK and Refused')
    print('Scenario denominator: valid responses only (exclude DK AND Refused)')
    print(f'Mean ABSOLUTE gap change across 24 countries: {mean_abs:.6f} pp')
    print(f'Mean SIGNED gap change across 24 countries:   {mean_signed:+.6f} pp')
    print(f'Largest individual change: {largest.country} {largest.change_pp:+.6f} pp')
    print('\nCountry sensitivity audit (pp; scenario minus baseline):')
    print(sav_audit[['country','baseline_gap_pp','exclude_dk_refused_gap_pp',
                     'change_pp']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    assert len(raw_sav) == len(prod_sav) == len(prod_csv) == EXPECTED_N
    assert prod_sav.id.is_unique
    print('\nPASS: 28,333 respondents retained; 185 unknown age bands remain unclassified')

    OUT.mkdir(parents=True, exist_ok=True)
    sav_audit.to_csv(OUT / 'task24_q30_denominator_country_audit.csv', index=False)
    sav_groups.to_csv(OUT / 'task24_q30_denominator_age_group_audit.csv', index=False)
    summary = pd.DataFrame([{
        'task': 24, 'scenario': 'exclude_dk_and_refused',
        'estimator': 'weighted Q30 A lot percentage 18-29 minus 65+ (country pp)',
        'countries': len(sav_audit), 'age_country_cells': len(sav_groups),
        'mean_absolute_change_pp': mean_abs,
        'mean_signed_change_pp': mean_signed,
        'largest_change_country': largest.country,
        'largest_change_pp': float(largest.change_pp),
        'total_respondents_preserved': EXPECTED_N,
        'known_age_band_n': 28_148, 'missing_age_band_n': 185,
        'validation_status': 'SAV_AND_CSV_MATCHED',
    }])
    summary.to_csv(OUT / 'task24_q30_denominator_summary.csv', index=False)
    pd.DataFrame([{
        'task': 24,
        'decision': 'Retain DK/refusals in Pew-style topline denominator for baseline; explicitly test excluding both from Q30 denominator',
        'alternative': 'Exclude both DK and Refused from all analytical denominators as a default',
        'reason': 'Pew topline is DK-inclusive; nonresponse handling affects percentages and country-by-age gaps',
        'effect': f'Absolute mean change of {mean_abs:.6f} percentage points in country 18-29 minus 65+ weighted Q30 A lot gaps; maximum individual {largest.country} {largest.change_pp:+.6f} pp',
        'limitations': 'Illustrative outcome only; P3/P4 analysis may specify another estimand. Small oldest groups can create unstable gaps.',
        'validation': 'Original SAV labels; original SAV/CSV by ID; independent age recovery; production gap calculation',
        'status': 'VERIFIED_WHEN_SCRIPT_PASSES',
    }]).to_csv(OUT / 'task24_decision_log_entry.csv', index=False)
    pd.DataFrame([{
        'task': 24,
        'ai_contribution': 'Generated weighted denominator sensitivity validator, respondent-level cross-source checks, independent numerator/denominator recomputation, and synthetic tests',
        'data_basis': 'Original SAV Q30 response meanings and respondent-level SAV/CSV; previous Task16 age categories',
        'independent_check': 'Separate calculations from raw response codes compared to src.cleaning.awareness_gap',
        'student_verification': 'Executing validate_task24.py on original SAV and reviewing generated audits',
        'ai_errors_detected': 'None detected in delivered Task24 code at time of release',
    }]).to_csv(OUT / 'task24_ai_use_log_entry.csv', index=False)
    (OUT / 'task24_validation_note.txt').write_text(
        'TASK 24 - Q30 NONRESPONSE DENOMINATOR SENSITIVITY\n'
        'Illustrative country-specific age gap: weighted P(Q30="A lot" | age 18-29) '
        'minus weighted P(Q30="A lot" | age 65+). Baseline uses everyone in '
        'each age group including DK/refusal in denominator (Pew topline convention). '
        'Sensitivity denominator omits BOTH SAV-labelled DK and Refused. '
        'Numerator, Q30 response codes and weights are taken from ORIGINAL raw data; '
        'age band comes from validated production cleaning and independently checked '
        'Task14/16 recovery. No respondents removed or original data overwritten.\n'
        f'Country gap mean absolute change: {mean_abs:.6f} percentage points. '
        f'Largest change: {largest.country}, {largest.change_pp:+.6f} pp. '
        'This is a descriptive sensitivity, not a formal survey-design variance '
        'or statistical significance test; it is not a DK-only scenario.\n',
        encoding='utf-8')
    print('\nSaved: task24_q30_denominator_country_audit.csv,')
    print('       task24_q30_denominator_age_group_audit.csv, task24_q30_denominator_summary.csv,')
    print('       task24_decision_log_entry.csv, task24_ai_use_log_entry.csv,')
    print('       task24_validation_note.txt')
    print('TASK 24 PASSED: excluding Q30 DK+Refused changes weighted country age gaps')
    print(f'                by {mean_abs:.6f} pp absolute mean; 24/24 SAV/CSV agree; no rows removed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--self-test', action='store_true')
    grp.add_argument('--csv-diagnostic', action='store_true')
    options = parser.parse_args()
    if options.self_test:
        self_test()
    elif options.csv_diagnostic:
        diagnostic()
    else:
        main()
