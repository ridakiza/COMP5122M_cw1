"""COMP5122M P1 — Task 21: age-by-country cell sizes and denominators.

From the repository ROOT:
    python scripts/validate_task21.py --self-test
    python scripts/validate_task21.py

Uses ORIGINAL SAV value labels to run the verified production cleaning pipeline
and independently reconstructs age bands via Tasks 14 and 16. Verifies
respondent-level SAV/CSV agreement and audits all 24 x 4 age-country cells.

IMPORTANT: n_age_band counts all respondents with an identifiable age band,
not those who necessarily answered Q30/Q31. The question-specific valid Ns
are separate. No significance test or arbitrary suppression rule is applied.
This validation never overwrites pew_clean.csv or the original input files.
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
    from src.cleaning import do_clean, country_names  # type: ignore
    from validate_task14 import country_age_columns, prepare, recover  # type: ignore
    from validate_task16 import make_age_band  # type: ignore
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Task 21 needs src/cleaning.py, validate_task14.py and validate_task16.py. "
        f"Missing import: {exc}. Keep their paths in the existing repository."
    ) from exc

SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
OUT = ROOT / 'data/clean'
BANDS = ['18-29', '30-49', '50-64', '65+']
EXPECTED = {
    'total': 28_333, 'countries': 24, 'known_age': 28_148,
    'missing_age': 185, 'recovered': 111, 'groups': 96,
    'n_lt_100': 9, 'n_lt_50': 3, 'n_lt_30': 1,
}


def norm_band(series: pd.Series) -> pd.Series:
    """Compare the typographic en dash in production with Tasks 14/16's hyphen."""
    return (series.astype('string').str.strip()
            .str.replace('–', '-', regex=False).str.replace('—', '-', regex=False))


def numeric(series: pd.Series) -> pd.Series:
    """Strictly parse numeric source values; whitespace = structural blank."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='raise')
    text = series.astype('string').str.strip()
    return pd.to_numeric(text.mask(text.eq('')), errors='raise')


def assert_same_by_id(left: pd.DataFrame, right: pd.DataFrame, fields: list[str],
                      *, context: str) -> None:
    """Order-independent respondent-level equality, including missing cells."""
    for frame in (left, right):
        assert 'id' in frame and frame.id.notna().all() and frame.id.is_unique, (
            f'{context}: invalid respondent IDs')
    a = left.set_index('id').sort_index()
    b = right.set_index('id').sort_index()
    assert a.index.equals(b.index), f'{context}: respondent IDs differ'
    for col in fields:
        if col == 'age_band':
            equal = norm_band(a[col]).fillna('<missing>').equals(
                norm_band(b[col]).fillna('<missing>'))
        elif col == 'age_band_recovered':
            equal = a[col].astype('bool').equals(b[col].astype('bool'))
        else:
            av = numeric(a[col]).to_numpy(dtype=float, na_value=np.nan)
            bv = numeric(b[col]).to_numpy(dtype=float, na_value=np.nan)
            equal = bool(np.isclose(av, bv, rtol=0, atol=1e-10,
                                    equal_nan=True).all())
        assert equal, f'{context}: respondent-level mismatch for {col}'
    print(f'PASS: {context} — {len(a):,} matched IDs, {len(fields)} fields')


def assert_independent_age(prod: pd.DataFrame, independent: pd.DataFrame,
                           *, context: str) -> None:
    """Verify both ages and recovery flags against a DIFFERENT implementation."""
    prod_index = prod.set_index('id').sort_index()
    ind_index = independent.set_index('id').sort_index()
    assert prod_index.index.equals(ind_index.index), f'{context}: independent respondent IDs differ'
    assert norm_band(prod_index.age_band).fillna('<NA>').equals(
        norm_band(ind_index.age_band).fillna('<NA>')), f'{context}: independent age bands differ'
    # pandas bool and nullable BooleanDtype may have identical 28,333 values
    # while .equals() returns False solely because the dtypes differ.
    production_flags = prod_index.age_band_recovered.to_numpy(dtype=bool)
    independent_flags = ind_index.age_band_source.eq('recovered_followup').to_numpy(dtype=bool)
    assert np.array_equal(production_flags, independent_flags), (
        f'{context}: independently recovered age flags differ')


def check_independent(raw: pd.DataFrame, prod: pd.DataFrame,
                      meta: object, *, name: str) -> None:
    """Age independently reconstructed from SAV labels; never reuse do_clean()."""
    _, age_fields = country_age_columns(meta)
    required = ['id', 'country', 'age', 'age_australia', *sorted(age_fields.values())]
    source = prepare(raw, required, name + ' independent age input')
    partial, _, _ = recover(source, meta, strict=True)
    independent = make_age_band(partial, meta, strict=True)
    assert_independent_age(prod, independent, context=name)
    print(f'PASS: {name} production vs independently reconstructed age bands — '
          f'{len(independent):,} respondents')


def group_audit(raw: pd.DataFrame, cleaned: pd.DataFrame, names: dict[int, str],
                *, strict: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all country x band cells, and country totals including age missingness."""
    required = {'id', 'country', 'age', 'age_band', 'age_band_recovered',
                'ai_heard', 'ai_cncexc', 'weight'}
    assert required.issubset(set(raw) | set(cleaned)), 'Missing age/country/Q30/Q31 fields'
    assert len(raw) == len(cleaned), 'Cleaning removed a respondent'
    assert numeric(raw.id).equals(numeric(cleaned.id)), 'Respondent order/IDs changed'
    assert cleaned.id.is_unique and cleaned.id.notna().all()
    assert numeric(raw.country).equals(numeric(cleaned.country)), 'Country changed'
    assert numeric(cleaned.weight).gt(0).all(), 'Invalid sampling weights'

    df = pd.DataFrame({
        'id': numeric(cleaned.id),
        'country_code': numeric(raw.country).astype(int),
        'age_band': norm_band(cleaned.age_band),
        'recovered': cleaned.age_band_recovered.astype(bool),
        'raw_age': numeric(raw.age),
        'q30_valid': numeric(cleaned.ai_heard).notna(),
        'q31_valid': numeric(cleaned.ai_cncexc).notna(),
    })
    assert set(df.country_code.unique()) == set(names), 'Unrecognised/missing country codes'
    assert df.age_band.dropna().isin(BANDS).all(), 'Unexpected age-band category'
    assert not (df.recovered & df.age_band.isna()).any(), 'A recovered age has no age band'
    assert df.loc[df.recovered, 'raw_age'].isin([98, 99]).all(), (
        'Recovery applied to someone with a known exact age')
    assert df.loc[df.recovered, 'country_code'].ne(
        next(k for k, label in names.items() if label.casefold() == 'australia')).all(), (
        'Incorrect Australian follow-up recovery')
    df['country'] = df.country_code.map(names)
    records = []
    countries = []
    for code, country in sorted(names.items(), key=lambda pair: pair[1]):
        subset = df.loc[df.country_code.eq(code)]
        missing_n = int(subset.age_band.isna().sum())
        recovered_n = int(subset.recovered.sum())
        known_n = int(subset.age_band.notna().sum())
        countries.append({
            'country_code': code, 'country': country, 'country_respondents': len(subset),
            'known_age_band_n': known_n, 'missing_age_band_n': missing_n,
            'recovered_age_band_n': recovered_n,
        })
        for band in BANDS:
            cell = subset.loc[subset.age_band.eq(band)]
            n = len(cell)
            rec = int(cell.recovered.sum())
            # 'without_recovery_n' includes Australian reported bands; the
            # exact-age-only comparison is separately meaningful for India.
            exact_n = int(cell.raw_age.between(18, 97).sum())
            australia_n = int(n - exact_n - rec)
            assert australia_n >= 0, 'Negative age-band provenance count'
            if country != 'Australia':
                assert australia_n == 0, f'{country} has an unexpected age source'
            else:
                assert exact_n == rec == 0, 'Australia should only use reported age bands'
            q30 = int(cell.q30_valid.sum())
            q31 = int(cell.q31_valid.sum())
            records.append({
                'country_code': code, 'country': country, 'age_band': band,
                'n_age_band': n, 'exact_age_n': exact_n,
                'followup_recovered_n': rec, 'australia_reported_n': australia_n,
                'n_without_followup_recovery': n - rec,
                'q30_valid_n': q30, 'q30_missing_n': n - q30,
                'q31_valid_n': q31, 'q31_missing_n': n - q31,
                'small_n_lt_100': n < 100,
                'small_n_lt_50': n < 50,
                'small_n_lt_30': n < 30,
            })
        assert sum(r['n_age_band'] for r in records[-4:]) + missing_n == len(subset), (
            f'{country}: age groups plus missing do not sum to country total')
        assert sum(r['followup_recovered_n'] for r in records[-4:]) == recovered_n
    groups = pd.DataFrame.from_records(records)
    totals = pd.DataFrame.from_records(countries)
    assert int(groups.n_age_band.sum() + totals.missing_age_band_n.sum()) == len(cleaned)
    assert int(groups.followup_recovered_n.sum()) == int(df.recovered.sum())
    assert (groups.exact_age_n + groups.followup_recovered_n
            + groups.australia_reported_n).equals(groups.n_age_band), 'Age source counts mismatch'
    assert (groups.q30_valid_n + groups.q30_missing_n).equals(groups.n_age_band)
    assert (groups.q31_valid_n + groups.q31_missing_n).equals(groups.n_age_band)

    if strict:
        assert len(cleaned) == EXPECTED['total'] and len(names) == EXPECTED['countries']
        assert len(groups) == EXPECTED['groups'], 'Expected 24 x 4 age-country cells'
        assert int(groups.n_age_band.sum()) == EXPECTED['known_age']
        assert int(totals.missing_age_band_n.sum()) == EXPECTED['missing_age']
        assert int(groups.followup_recovered_n.sum()) == EXPECTED['recovered']
        for field, expected in [('small_n_lt_100', 'n_lt_100'),
                                ('small_n_lt_50', 'n_lt_50'),
                                ('small_n_lt_30', 'n_lt_30')]:
            assert int(groups[field].sum()) == EXPECTED[expected], (
                f'Unexpected number of age-country cells with {field}: '
                f'{int(groups[field].sum())}')
        young = groups.loc[groups.age_band.eq('18-29')]
        assert int(young.n_age_band.min()) == 80
        assert set(young.loc[young.n_age_band.eq(80), 'country']) == {'Hungary', 'South Korea'}
        india = young.loc[young.country.eq('India')].iloc[0]
        assert int(young.n_age_band.max()) == int(india.n_age_band) == 1_085
        assert (int(india.exact_age_n) == 1_083
                and int(india.followup_recovered_n) == 2
                and int(india.n_without_followup_recovery) == 1_083)
        assert int(groups.n_age_band.min()) == 20
        small = groups.loc[groups.n_age_band.eq(20), ['country', 'age_band']]
        assert small.to_dict('records') == [{'country': 'Nigeria', 'age_band': '65+'}]
    return groups, totals


def self_test() -> None:
    """Synthetic cases test counts, provenance, missing respondents and corruption."""
    raw = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7],
        'country': [1, 1, 1, 1, 2, 2, 2],
        'age': [18, 99, 54, 98, np.nan, np.nan, np.nan],
        'weight': [1.0] * 7,
    })
    clean = pd.DataFrame({
        'id': raw.id, 'country': raw.country, 'weight': raw.weight,
        'age_band': ['18–29', '18–29', '50–64', None, '18–29', '65+', None],
        'age_band_recovered': [False, True, False, False, False, False, False],
        'ai_heard': [1, np.nan, 2, 2, 1, 3, 3],
        'ai_cncexc': [1, 3, np.nan, 2, 3, 1, 2],
    })
    names = {1: 'India', 2: 'Australia'}
    groups, totals = group_audit(raw, clean, names, strict=False)
    india_young = groups.loc[groups.country.eq('India') & groups.age_band.eq('18-29')].iloc[0]
    assert (india_young.n_age_band, india_young.exact_age_n,
            india_young.followup_recovered_n, india_young.q30_valid_n) == (2, 1, 1, 1)
    assert totals.loc[totals.country.eq('Australia'), 'missing_age_band_n'].item() == 1
    assert len(groups) == 8 and groups.n_age_band.sum() == 5
    print('PASS: age-band cell Ns include recovered ages and Australian age-band responses')
    print('PASS: question-specific valid Ns exclude nonresponse without dropping people')
    indep = pd.DataFrame({
        'id': clean.id,
        'age_band': ['18-29', '18-29', '50-64', None, '18-29', '65+', None],
        'age_band_source': ['exact_age', 'recovered_followup', 'exact_age',
                            'unclassified', 'australia_reported',
                            'australia_reported', 'australia_missing'],
    })
    assert_independent_age(clean, indep, context='self-test correct recovery')
    bad = clean.copy(deep=True)
    bad.loc[bad.id.eq(2), 'age_band_recovered'] = False
    try:
        assert_independent_age(bad, indep, context='self-test wrong recovered flag')
    except AssertionError:
        print('PASS: omitted age-recovery provenance detected by independent classifier')
    else:
        raise AssertionError('Failed to catch omitted age-recovery provenance')
    # Test corruption that MUST fail regardless of the independent classifier.
    bad = clean.copy(deep=True)
    bad.loc[bad.id.eq(2), 'age_band'] = 'unknown'
    try:
        group_audit(raw, bad, names, strict=False)
    except AssertionError:
        print('PASS: unexpected age group rejected')
    else:
        raise AssertionError('Failed to reject unexpected age group')
    mismatch = clean.copy(deep=True)
    mismatch.loc[mismatch.id.eq(2), 'age_band'] = '30–49'
    try:
        assert_same_by_id(clean, mismatch, ['age_band'], context='self-test wrong age')
    except AssertionError:
        print('PASS: respondent-level age-band corruption detected')
    else:
        raise AssertionError('Failed to detect respondent-level corruption')
    assert_same_by_id(clean, clean.sample(frac=1, random_state=23),
                      ['age_band', 'age_band_recovered', 'ai_heard'], context='self-test row reorder')
    print('TASK 21 SELF-TEST PASSED')


def main() -> None:
    try:
        import pyreadstat
    except ImportError as exc:
        raise RuntimeError('Install project requirements (including pyreadstat) in your venv') from exc
    for path in (SAV, CSV):
        if not path.is_file():
            raise FileNotFoundError(f'Missing original file: {path}')

    print('=== TASK 21: AGE-COUNTRY SUBGROUP SIZES (FINAL RECOVERED AGE BANDS) ===')
    raw_sav, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    raw_csv = pd.read_csv(CSV, encoding='utf-8-sig', low_memory=False)
    raw_sav.columns = raw_sav.columns.str.strip().str.lower()
    raw_csv.columns = raw_csv.columns.str.strip().str.lower()
    assert raw_sav.shape == raw_csv.shape == (28_333, 483), 'Unexpected raw dataset shape'
    names = country_names(meta)
    assert len(names) == 24
    sav_before = raw_sav.copy(deep=True)
    csv_before = raw_csv.copy(deep=True)
    cleaned_sav = do_clean(raw_sav, meta, log=[])
    cleaned_csv = do_clean(raw_csv, meta, log=[])
    pd.testing.assert_frame_equal(raw_sav, sav_before, check_exact=True)
    pd.testing.assert_frame_equal(raw_csv, csv_before, check_exact=True)
    print('PASS: SAV and CSV original files unchanged by production cleaning')

    assert_same_by_id(cleaned_sav, cleaned_csv,
                      ['country', 'age_band', 'age_band_recovered',
                       'ai_heard', 'ai_cncexc', 'weight'], context='SAV vs CSV after cleaning')
    check_independent(raw_sav, cleaned_sav, meta, name='SAV')
    check_independent(raw_csv, cleaned_csv, meta, name='CSV')
    groups_sav, countries_sav = group_audit(raw_sav, cleaned_sav, names)
    groups_csv, countries_csv = group_audit(raw_csv, cleaned_csv, names)
    pd.testing.assert_frame_equal(groups_sav, groups_csv, check_exact=True)
    pd.testing.assert_frame_equal(countries_sav, countries_csv, check_exact=True)
    print('PASS: all 96 SAV/CSV age-country cell counts agree, including Q30/Q31 valid Ns')

    young = groups_sav.loc[groups_sav.age_band.eq('18-29')]
    print('\nAges 18–29: country sample sizes (including recovered follow-ups):')
    print(young[['country', 'n_age_band', 'exact_age_n', 'followup_recovered_n']]
          .sort_values(['n_age_band', 'country']).to_string(index=False))
    india = young.loc[young.country.eq('India')].iloc[0]
    print(f'\nIndia 18–29: exact ages only = {india.exact_age_n:,}; '
          f'+ {india.followup_recovered_n:,} recovered = {india.n_age_band:,} final')
    print('Smallest 18–29 cells: Hungary=80, South Korea=80; largest: India=1,085')
    flagged = groups_sav.loc[groups_sav.small_n_lt_100].sort_values(
        ['n_age_band', 'country', 'age_band'])
    print('\nAll age-country cells with fewer than 100 respondents:')
    print(flagged[['country', 'age_band', 'n_age_band', 'q30_valid_n',
                   'q31_valid_n']].to_string(index=False))
    print(f'\nCell sizes: {len(groups_sav)} cells; 9 below 100; 3 below 50; 1 below 30.')
    print('Smallest across ALL age groups: Nigeria ages 65+ = 20.')
    print(f'Known age bands: {groups_sav.n_age_band.sum():,}; '
          f'missing: {countries_sav.missing_age_band_n.sum():,}; '
          f'all respondents: {len(cleaned_sav):,}.')
    print('NOTE: age-group n includes Q30/Q31 nonrespondents. Question-specific valid Ns are separate.')

    OUT.mkdir(parents=True, exist_ok=True)
    groups_sav.to_csv(OUT / 'task21_country_age_cells.csv', index=False)
    countries_sav.to_csv(OUT / 'task21_country_age_missing_audit.csv', index=False)
    young.sort_values(['n_age_band', 'country']).to_csv(
        OUT / 'task21_young_sample_comparison.csv', index=False)
    flagged.to_csv(OUT / 'task21_small_cell_flags.csv', index=False)
    pd.DataFrame([{
        'task': 21,
        'decision': 'Count all respondents in each final 24-country x 4-age-band cell, including verified follow-up recoveries',
        'alternative': 'Use only respondents with numeric exact ages; or use question-specific valid N as age-group size',
        'reason': 'Task16 included 111 defensible age-band recoveries and 1,827 Australian age reports; question nonresponse affects per-question N, not subgroup membership',
        'effect': 'India ages 18-29: 1083 exact-age-only vs 1085 final; Hungary/South Korea young n=80; 9 age-country cells have n<100; 185 missing age bands',
        'statistical_caveat': 'Small unweighted cell Ns reduce precision; survey weighting/design can reduce effective N further; no universal n cutoff imposed',
        'verification': 'Independent Task14/16 reconstruction and production pipeline; SAV/CSV matched by ID and in all 96 cells',
        'status': 'VERIFIED',
    }]).to_csv(OUT / 'task21_decision_log_entry.csv', index=False)
    pd.DataFrame([{
        'task': 21,
        'ai_contribution': 'Generated age-country cell audit, question-specific denominators, independent cross-checks and synthetic tests',
        'verification': 'Original SAV and CSV, SAV age labels, independent Task14/16 age reconstruction, main production cleaning',
        'human_verification': 'Student runs python scripts/validate_task21.py against original SAV',
        'ai_error_or_limitation': 'Earlier 1083 India young count is exact-age-only; initial Task21 test also incorrectly used pandas Series.equals() across bool and nullable-Boolean dtypes',
        'correction': 'Two recovered Indian age bands raise young N to 1085; check identical Boolean values using np.array_equal rather than requiring identical pandas dtypes',
    }]).to_csv(OUT / 'task21_ai_use_log_entry.csv', index=False)
    pd.DataFrame([
        {
            'task': 21, 'issue': 'India ages 18-29 sample size cited without recovery qualifier',
            'original_issue': 'Unqualified exact-age-only count n=1083',
            'why': 'Two unambiguous Indian age bands recovered in Task14',
            'detection': 'Compare exact-age-only vs final Task16 groups',
            'resolution': 'Report n=1085 for final age group; n=1083 exact-age-only baseline',
            'effect_on_data': 'None; reporting correction only',
            'status': 'corrected',
        }, {
            'task': 21, 'issue': 'AI-generated validator compared equivalent bool and nullable Boolean arrays with Series.equals',
            'original_issue': 'False-positive assertion despite zero recovery-flag mismatches',
            'why': 'Series.equals requires matching dtypes; production bool vs independent pandas BooleanDtype',
            'detection': 'Local full-CSV diagnostic: 111 flags in each, np.array_equal true, Series.equals false',
            'resolution': 'Compare dtype-normalised bool arrays using np.array_equal; rerun synthetic and full CSV checks',
            'effect_on_data': 'None; test-code-only issue found before delivery',
            'status': 'corrected',
        },
    ]).to_csv(OUT / 'task21_ai_discrepancy_entry.csv', index=False)
    (OUT / 'task21_validation_note.txt').write_text(
        'Task 21: age-band subgroup Ns count everyone with a known final age band, '
        'not only Q30/Q31 valid respondents. All 96 country/age cells audited. '
        'Hungary and South Korea 18-29 n=80 each; India 18-29 n=1085 '
        '(1083 exact ages plus 2 recovered follow-up bands). '
        '9 cells n<100, 3 n<50, 1 n<30; smallest Nigeria 65+ n=20. '
        'No arbitrary n threshold or precision claim is imposed; interpret small cells cautiously. '
        '28,148 age-known, 185 missing, all 28,333 rows retained. '
        'SAV/CSV and independent Task14/16 age classifications agree by respondent ID.\n',
        encoding='utf-8'
    )
    print('\nSaved: task21_country_age_cells.csv, task21_country_age_missing_audit.csv,')
    print('       task21_young_sample_comparison.csv, task21_small_cell_flags.csv,')
    print('       task21_decision_log_entry.csv, task21_ai_use_log_entry.csv,')
    print('       task21_ai_discrepancy_entry.csv, task21_validation_note.txt')
    print('TASK 21 PASSED: 96/96 age-country cells audited; 80–1,085 young respondents,')
    print('                9 cells below 100; SAV/CSV and independent recovery agree.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate age-country subgroup cell sizes')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
