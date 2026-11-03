"""COMP5122M P1 Task 22 — WEIGHT-ONLY (Kish) EFFECTIVE SAMPLE SIZE.

Run from the repository root:
    python scripts/validate_task22.py --self-test
    python scripts/validate_task22.py

Optional diagnostic without the SAV (NOT a substitute for final SAV verification):
    python scripts/validate_task22.py --csv-diagnostic

The Kish weight-only approximation for each country is:
    n_eff,w = (sum_i w_i)^2 / sum_i(w_i^2)
    DEFF_w  = n / n_eff,w = 1 + CV_population(w)^2

It is NOT the full survey design effect: clustering, stratification, sample
selection, survey adjustment, and estimator-specific variance are not included.
The original SAV/CSV, and the production pew_clean.csv, are never modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
OUT = ROOT / 'data/clean'
FIELDS = ['id', 'country', 'weight']
EXPECTED_N = 28_333
EXPECTED_COUNTRIES = 24
COUNTRY_SUM_TOL = 1e-6
SOURCE_TOL = 1e-10

# Approximate reference values independently checked from the original CSV.
# These are validation targets, NOT coefficients used to calculate DEFF_w.
EXPECTED_SPOT = {
    'Italy': (1007, 2.17541, 462.90049),
    'India': (3545, 2.06761, 1714.53892),
    'Australia': (1829, 1.21284, 1508.03090),
}


def normalise_source(source: pd.DataFrame, name: str) -> pd.DataFrame:
    """Strict numeric checks, work on a copy, and preserve respondent IDs."""
    frame = source.copy(deep=True)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    assert frame.columns.is_unique, f'{name}: duplicate column names'
    assert set(FIELDS).issubset(frame.columns), f'{name}: missing {set(FIELDS)-set(frame.columns)}'
    frame = frame.loc[:, FIELDS].copy()
    for col in FIELDS:
        parsed = pd.to_numeric(frame[col], errors='coerce')
        assert parsed.notna().all(), f'{name}: missing/non-numeric {col}'
        assert np.isfinite(parsed.to_numpy(dtype=float)).all(), f'{name}: nonfinite {col}'
        frame[col] = parsed
    for col in ('id', 'country'):
        assert (frame[col] == np.floor(frame[col])).all(), f'{name}: nonintegral {col}'
        frame[col] = frame[col].astype('int64')
    assert frame['id'].is_unique, f'{name}: duplicate respondent IDs'
    assert frame['weight'].gt(0).all(), f'{name}: zero/negative survey weights'
    return frame


def compare_by_id(sav: pd.DataFrame, csv: pd.DataFrame) -> float:
    """Reject individual mismatches even if country totals still agree."""
    a = normalise_source(sav, 'SAV').set_index('id').sort_index()
    b = normalise_source(csv, 'CSV').set_index('id').sort_index()
    assert a.index.equals(b.index), 'SAV/CSV respondent ID sets differ'
    assert a.country.equals(b.country), 'SAV/CSV country ownership differs'
    wa = a.weight.to_numpy(dtype=float)
    wb = b.weight.to_numpy(dtype=float)
    assert np.isclose(wa, wb, atol=SOURCE_TOL, rtol=0).all(), 'SAV/CSV individual weights differ'
    return float(np.max(np.abs(wa - wb)))


def weight_kish_country(frame: pd.DataFrame, name: str,
                        expected_n: int | None = EXPECTED_N,
                        expected_countries: int | None = EXPECTED_COUNTRIES,
                        require_normalised: bool = True) -> pd.DataFrame:
    """Compute BOTH sum-squares and independently the population-CV forms."""
    df = normalise_source(frame, name)
    if expected_n is not None:
        assert len(df) == expected_n, f'{name}: expected {expected_n} people, got {len(df)}'
    if expected_countries is not None:
        assert df.country.nunique() == expected_countries, f'{name}: unexpected country count'

    records: list[dict] = []
    for country, subset in df.groupby('country', sort=True):
        w = subset.weight.to_numpy(dtype=np.float64)
        n = len(w)
        sw = float(np.sum(w, dtype=np.float64))
        sw2 = float(np.sum(np.square(w), dtype=np.float64))
        mean_w = float(np.mean(w))
        # 1+CV^2 is algebraically independent of the sum-of-squares ratio.
        pop_cv = float(np.std(w, ddof=0) / mean_w)
        neff = sw**2 / sw2
        deff = n / neff
        cv_deff = 1.0 + pop_cv**2
        assert np.isclose(deff, cv_deff, atol=1e-11, rtol=0), (
            f'{name}: DEFF_w vs 1+CV^2 identity fails for country code {country}')
        assert 1 - 1e-11 <= deff and 0 < neff <= n + 1e-8, (
            f'{name}: impossible weight-only effective sample size for {country}')
        normalised = abs(sw - n) <= COUNTRY_SUM_TOL
        if require_normalised:
            assert normalised, f'{name}: weights do not sum to N for country {country}'
        records.append({
            'country_code': int(country),
            'raw_n': n,
            'sum_weights': sw,
            'sum_weights_squared': sw2,
            'mean_weight': mean_w,
            'min_weight': float(w.min()),
            'max_weight': float(w.max()),
            'population_cv_weights': pop_cv,
            'weight_only_deff': deff,
            'kish_effective_n': neff,
            'effective_n_pct_of_raw': 100 * neff / n,
            'effective_n_reduction': n - neff,
            'independent_cv_deff': cv_deff,
            'country_weights_normalised': normalised,
        })
    result = pd.DataFrame.from_records(records)
    assert int(result.raw_n.sum()) == len(df), f'{name}: country sizes do not sum to total'
    if require_normalised:
        assert abs(float(result.sum_weights.sum()) - len(df)) <= COUNTRY_SUM_TOL
    return result


def country_name_labels(meta: object, country_codes: set[int]) -> dict[int, str]:
    labels_all = getattr(meta, 'variable_value_labels', {}) or {}
    matching = [v for k, v in labels_all.items() if str(k).strip().lower() == 'country']
    assert len(matching) == 1, 'SAV missing unique country code/value labels'
    labels = {int(float(k)): str(v).strip() for k, v in matching[0].items()}
    assert set(labels) == country_codes, 'SAV country labels do not match data codes'
    assert len(set(labels.values())) == len(labels), 'Duplicate country labels in SAV'
    assert {'Italy', 'India', 'Australia'}.issubset(set(labels.values())), (
        'Expected spot-check country labels not found')
    return labels


def check_spots(df: pd.DataFrame) -> None:
    for name, (expected_n, expected_deff, expected_neff) in EXPECTED_SPOT.items():
        row = df.loc[df.country.eq(name)]
        assert len(row) == 1, f'Missing country: {name}'
        row = row.iloc[0]
        assert int(row.raw_n) == expected_n, f'{name}: raw sample size changed'
        assert abs(row.weight_only_deff - expected_deff) < 0.00001, (
            f'{name}: unexpected weight-only DEFF')
        assert abs(row.kish_effective_n - expected_neff) < 0.00005, (
            f'{name}: unexpected Kish effective N')


def assert_country_agreement(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Check all 24 estimates, not just 3 reference countries."""
    a = left.set_index('country_code').sort_index()
    b = right.set_index('country_code').sort_index()
    assert a.index.equals(b.index), 'SAV/CSV country codes disagree'
    cols = ['raw_n', 'sum_weights', 'sum_weights_squared', 'mean_weight',
            'min_weight', 'max_weight', 'population_cv_weights',
            'weight_only_deff', 'kish_effective_n', 'effective_n_pct_of_raw']
    for col in cols:
        assert np.isclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float),
                          atol=1e-8, rtol=0).all(), f'SAV/CSV mismatch for {col}'


def must_fail(action, label: str) -> None:
    try:
        action()
    except (AssertionError, ValueError):
        print(f'PASS: {label}')
    else:
        raise AssertionError(f'Self-test did not detect: {label}')


def self_test() -> None:
    # Country A: 2 respondents, weights [0.5, 1.5]. n_eff=1.6, DEFF_w=1.25.
    # Country B: constant weights => DEFF_w=1 and n_eff=2.
    toy = pd.DataFrame({'id': [1, 2, 3, 4], 'country': [1, 1, 2, 2],
                        'weight': [0.5, 1.5, 1.0, 1.0]})
    original = toy.copy(deep=True)
    a = weight_kish_country(toy, 'TOY', expected_n=4, expected_countries=2)
    a1 = a.set_index('country_code').loc[1]
    a2 = a.set_index('country_code').loc[2]
    assert np.isclose(a1.kish_effective_n, 1.6)
    assert np.isclose(a1.weight_only_deff, 1.25)
    assert np.isclose(a2.kish_effective_n, 2.0)
    assert np.isclose(a2.weight_only_deff, 1.0)
    pd.testing.assert_frame_equal(toy, original, check_exact=True)
    print('PASS: hand-calculated Kish effective N and 1+CV² formula; input unchanged')

    shuffle = toy.sample(frac=1, random_state=22)
    assert np.isclose(compare_by_id(toy, shuffle), 0.0)
    print('PASS: SAV/CSV comparison independent of row order')
    bad = toy.copy()
    bad.loc[0, 'weight'] = 0.0
    must_fail(lambda: weight_kish_country(bad, 'TOY', 4, 2), 'zero weight rejected')
    bad.loc[0, 'weight'] = np.nan
    must_fail(lambda: weight_kish_country(bad, 'TOY', 4, 2), 'missing weight rejected')
    bad = toy.copy()
    bad.loc[0, 'weight'] = 0.7
    must_fail(lambda: weight_kish_country(bad, 'TOY', 4, 2), 'non-normalised country rejected')
    bad = toy.copy()
    bad.loc[0, 'weight'] = 0.6
    bad.loc[1, 'weight'] = 1.4
    must_fail(lambda: compare_by_id(toy, bad),
              'per-person weight mismatch caught even when country sums are identical')
    bad = toy.iloc[:-1].copy()
    must_fail(lambda: weight_kish_country(bad, 'TOY', 4, 2), 'missing respondent rejected')
    bad = toy.copy()
    bad.loc[1, 'id'] = 1
    must_fail(lambda: weight_kish_country(bad, 'TOY', 4, 2), 'duplicate respondent ID rejected')
    print('TASK 22 SELF-TEST PASSED')


def csv_diagnostic() -> None:
    assert CSV.exists(), f'CSV not found: {CSV}'
    raw = pd.read_csv(CSV, usecols=FIELDS, low_memory=False)
    audit = weight_kish_country(raw, 'CSV')
    assert len(audit) == EXPECTED_COUNTRIES
    print('=== TASK 22: CSV-ONLY DIAGNOSTIC (NOT SAV VERIFIED) ===')
    print(audit[['country_code', 'raw_n', 'weight_only_deff',
                 'kish_effective_n']].to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    print('CSV-ONLY DIAGNOSTIC PASSED; run without --csv-diagnostic for final SAV/CSV verification.')


def main() -> None:
    for path in (SAV, CSV):
        if not path.exists():
            raise FileNotFoundError(f'Missing original dataset: {path}')
    try:
        import pyreadstat
    except ImportError as exc:
        raise RuntimeError('Install pyreadstat in your venv: python -m pip install pyreadstat') from exc
    print('=== TASK 22: KISH WEIGHT-ONLY DESIGN-EFFECT APPROXIMATION ===')
    raw_sav, meta = pyreadstat.read_sav(str(SAV), usecols=FIELDS,
                                       apply_value_formats=False, user_missing=True)
    raw_csv = pd.read_csv(CSV, usecols=FIELDS, low_memory=False, encoding='utf-8-sig')
    original_sav = raw_sav.copy(deep=True)
    original_csv = raw_csv.copy(deep=True)
    assert raw_sav.shape == raw_csv.shape == (EXPECTED_N, 3), 'Unexpected source dimensions'
    max_abs = compare_by_id(raw_sav, raw_csv)
    print(f'PASS: SAV/CSV match for {EXPECTED_N:,} IDs, countries and individual weights '
          f'(max |weight difference| = {max_abs:.3g})')

    sav_audit = weight_kish_country(raw_sav, 'SAV')
    csv_audit = weight_kish_country(raw_csv, 'CSV')
    assert_country_agreement(sav_audit, csv_audit)
    print('PASS: 24/24 country Kish estimates match independently across SAV and CSV')
    labels = country_name_labels(meta, set(sav_audit.country_code))
    sav_audit.insert(1, 'country', sav_audit.country_code.map(labels))
    csv_audit.insert(1, 'country', csv_audit.country_code.map(labels))
    check_spots(sav_audit)
    check_spots(csv_audit)
    pd.testing.assert_frame_equal(raw_sav, original_sav, check_exact=True)
    pd.testing.assert_frame_equal(raw_csv, original_csv, check_exact=True)
    print(f'PASS: {EXPECTED_N:,} respondents preserved; original SAV/CSV DataFrames unchanged')
    print('\nSpot checks (weight-only approximations, NOT full survey-design DEFF):')
    spots = sav_audit.loc[sav_audit.country.isin(EXPECTED_SPOT)]
    print(spots[['country', 'raw_n', 'weight_only_deff', 'kish_effective_n',
                 'effective_n_pct_of_raw']].to_string(index=False,
                                                     float_format=lambda x: f'{x:.4f}'))
    print('\nAll 24 countries (descending weight-only DEFF):')
    ranked = sav_audit.sort_values('weight_only_deff', ascending=False)
    print(ranked[['country', 'raw_n', 'weight_only_deff', 'kish_effective_n']]
          .to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('\nCAUTION: Kish n_eff and DEFF_w use variation in weights ONLY. They are not')
    print('actual sampling variances, country confidence intervals, or a full complex-survey')
    print('design effect. The latter requires design information and is estimator-specific.')

    OUT.mkdir(parents=True, exist_ok=True)
    sav_audit['source'] = 'SAV_verified_against_CSV'
    sav_audit.sort_values('country').to_csv(OUT / 'task22_kish_weight_country_audit.csv', index=False)
    pd.DataFrame([{
        'task': 22, 'n_respondents': EXPECTED_N,
        'n_countries': len(sav_audit),
        'min_weight_only_deff': float(sav_audit.weight_only_deff.min()),
        'max_weight_only_deff': float(sav_audit.weight_only_deff.max()),
        'min_deff_country': str(sav_audit.loc[sav_audit.weight_only_deff.idxmin(), 'country']),
        'max_deff_country': str(sav_audit.loc[sav_audit.weight_only_deff.idxmax(), 'country']),
        'italy_weight_deff': float(spots.loc[spots.country.eq('Italy'), 'weight_only_deff'].iloc[0]),
        'india_weight_deff': float(spots.loc[spots.country.eq('India'), 'weight_only_deff'].iloc[0]),
        'australia_weight_deff': float(spots.loc[spots.country.eq('Australia'), 'weight_only_deff'].iloc[0]),
        'all_weight_sums_equal_country_n': bool(sav_audit.country_weights_normalised.all()),
        'all_24_sav_csv_match': True,
        'full_complex_design_deff_estimated': False,
    }]).to_csv(OUT / 'task22_weight_deff_summary.csv', index=False)
    pd.DataFrame([{
        'task': 22,
        'decision': 'Calculate per-country Kish n_eff and weight-only DEFF_w from original positive survey weights',
        'alternative': 'Assume n_eff equals raw N or claim this is the full survey design effect',
        'reason': 'Unequal weights reduce effective sample size in the weight-only approximation; clustering and stratification cannot be estimated from weights alone',
        'formula': 'n_eff=(sum w)^2/sum(w^2); DEFF_w=n/n_eff=1+population_CV(w)^2',
        'effect': 'Italy DEFF_w~2.18; India ~2.07; Australia ~1.21; no data or sample membership changed',
        'limitation': 'Weight-only Kish approximation is NOT actual complex-design DEFF, estimator-specific variance, or design-based CI',
        'status': 'VERIFIED_AGAINST_SAV_AND_CSV',
    }]).to_csv(OUT / 'task22_decision_log_entry.csv', index=False)
    pd.DataFrame([{
        'task': 22,
        'ai_contribution': 'Developed independent Kish weight-only effect calculator, SAV/CSV checks, alternative algebraic identity checks and synthetic error tests',
        'verification': 'Original SAV individual weights and labels; CSV respondent-by-respondent, 24 country audits and independent 1+CV² formula',
        'human_verification': 'Student executed validate_task22.py against original SAV',
        'ai_error_or_limitation': 'Do not describe weight-only DEFF as the full Pew sampling design effect',
        'correction': 'Label all metrics explicitly as Kish weight-only approximations',
    }]).to_csv(OUT / 'task22_ai_use_log_entry.csv', index=False)
    (OUT / 'task22_validation_note.txt').write_text(
        'Task 22: Kish weight-only design-effect APPROXIMATION by country.\n'
        'n_eff,w = (sum w)^2 / sum w^2; DEFF_w = n/n_eff,w = 1+population_CV(w)^2.\n'
        'Computed on each country\'s full original survey respondents, not on Q30/Q31 nonresponse-filtered records.\n'
        'Positive, complete, country-normalised weights verified; SAV and CSV match respondent by respondent.\n'
        'Reference country checks: Italy ~2.18, India ~2.07, Australia ~1.21.\n'
        'No full complex-sample design effect has been estimated: clustering, stratification and estimator-specific variance are not captured.\n'
        'Do not interpret Kish n_eff as a guaranteed precision value for an estimate; do not use it as a substitute for supplied survey design variables.\n'
        'No original survey data were changed and no respondents were excluded.\n',
        encoding='utf-8'
    )
    print('\nSaved: task22_kish_weight_country_audit.csv, task22_weight_deff_summary.csv,')
    print('       task22_decision_log_entry.csv, task22_ai_use_log_entry.csv,')
    print('       task22_validation_note.txt')
    print('TASK 22 PASSED: 24/24 country Kish weight-only approximations verified by SAV/CSV;')
    print('                Italy ~2.18, India ~2.07, Australia ~1.21; no rows changed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 22 Kish weight-only effective N and DEFF checks')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--csv-diagnostic', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.csv_diagnostic:
        csv_diagnostic()
    else:
        main()
