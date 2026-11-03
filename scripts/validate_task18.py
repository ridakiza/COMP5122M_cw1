"""COMP5122M P1 — Task 18: integrated post-cleaning validation.

From the REPOSITORY ROOT:
    python scripts/validate_task18.py --self-test
    python scripts/validate_task18.py

Validates *actual* src.cleaning.do_clean() on independent SAV/CSV inputs,
then compares age and gender against the separate, previously verified Task
14/16/17 implementations by respondent ID.  Does not save a cleaned dataset
or overwrite pew_clean.csv.  Audit CSVs are metadata/aggregate-only and
remain private under data/clean/.

CAUTION: raw d_educ_* columns are deliberately retained without recoding
until P2 supplies educ3.  'No missing codes left' applies to CLEANED ANALYTIC
variables, not preserved source columns or the pending education mapping.
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

from src.cleaning import do_clean, missing_codes, selected_columns  # type: ignore
from validate_task14 import country_age_columns, prepare, recover  # type: ignore
from validate_task16 import make_age_band  # type: ignore
from validate_task17 import clean_gender as independently_clean_gender  # type: ignore

SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
OUT = ROOT / 'data/clean'

N = 28_333
EXPECTED_BANDS = {'18-29': 6068, '30-49': 9787, '50-64': 6596, '65+': 5697}
EXPECTED_SOURCES = {
    'exact_age': 26210,
    'recovered_followup': 111,
    'australia_reported': 1827,
    'unclassified': 183,
    'australia_missing': 2,
}
EXPECTED_GENDER = {'Male': 15107, 'Female': 13203, 'Other': 21, 'Missing': 2}
EXPECTED_RECODED = {
    'ai_heard': 808, 'ai_cncexc': 1964,
    'aireg_us': 362, 'aireg_china': 380,
    'aireg_eu': 581, 'aireg_country': 213,
    'internetfreq': 94, 'age': 294,
}


def num(series: pd.Series) -> pd.Series:
    """Convert numeric codes without conflating blanks and response codes."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='raise')
    s = series.astype('string').str.strip()
    return pd.to_numeric(s.mask(s.eq('')), errors='raise')


def normalized_bands(series: pd.Series) -> pd.Series:
    """Make Task16's '18-29' and pipeline's '18–29' comparable."""
    return (series.astype('string').str.replace('–', '-', regex=False)
            .str.replace('—', '-', regex=False).str.strip())


def compare_columns_by_id(a: pd.DataFrame, b: pd.DataFrame, columns: list[str], *, stage: str) -> None:
    assert a.id.notna().all() and b.id.notna().all(), f'{stage}: IDs missing'
    assert a.id.is_unique and b.id.is_unique, f'{stage}: duplicate IDs'
    x = a.set_index('id').sort_index()
    y = b.set_index('id').sort_index()
    assert x.index.equals(y.index), f'{stage}: respondent IDs differ'
    for col in columns:
        if col == 'age_band':
            left, right = normalized_bands(x[col]), normalized_bands(y[col])
            equal = left.fillna('<NA>').equals(right.fillna('<NA>'))
        elif col in ('gender_label',):
            left = x[col].astype('string').fillna('<NA>')
            right = y[col].astype('string').fillna('<NA>')
            equal = left.equals(right)
        elif col == 'age_band_recovered':
            equal = x[col].astype(bool).equals(y[col].astype(bool))
        else:
            left = num(x[col]).to_numpy(dtype=float, na_value=np.nan)
            right = num(y[col]).to_numpy(dtype=float, na_value=np.nan)
            equal = bool(np.isclose(left, right, rtol=0, atol=1e-10, equal_nan=True).all())
        assert equal, f'{stage}: respondent-level mismatch in {col}'
    print(f'PASS: {stage} — {len(x):,} matched IDs, {len(columns)} columns')


def independent_age_gender(raw: pd.DataFrame, meta: object) -> pd.DataFrame:
    """Recalculate through DIFFERENT task validators, not do_clean()."""
    _, age_fields = country_age_columns(meta)
    wanted = ['id', 'country', 'age', 'age_australia', *sorted(age_fields.values())]
    age_input = prepare(raw, wanted, 'independent age check')
    partial, _, _ = recover(age_input, meta, strict=True)
    age = make_age_band(partial, meta, strict=True)
    gender = independently_clean_gender(raw, meta, strict=True)
    return age[['id', 'age_clean', 'age_band']].merge(
        gender[['id', 'gender_clean', 'gender_label']], on='id', validate='one_to_one'
    )


def check_postclean(raw: pd.DataFrame, clean: pd.DataFrame, meta: object,
                    source: str) -> tuple[dict, list[dict], list[dict]]:
    """Assert the prior task results and codebook-derived missing semantics."""
    assert raw.shape == (N, 483), f'{source}: wrong source shape'
    assert len(clean) == N, f'{source}: cleaned data lost rows'
    assert clean.id.equals(raw.id), f'{source}: row order or IDs changed'
    assert clean.id.notna().all() and clean.id.is_unique
    assert set(selected_columns(raw)).issubset(clean.columns), f'{source}: missing Part I variables'
    assert raw['country'].equals(clean['country']), f'{source}: country changed'
    assert np.isclose(num(raw.weight), num(clean.weight), rtol=0, atol=1e-10).all(), (
        f'{source}: survey weights changed'
    )

    # These are intentionally carried across unchanged for P2 / traceability.
    unchanged = ['gender', 'gender_australia', 'age_australia',
                 *[c for c in raw if c.startswith('d_educ_')]]
    for c in unchanged:
        a, b = num(raw[c]), num(clean[c])
        assert bool(np.isclose(a.to_numpy(dtype=float, na_value=np.nan),
                               b.to_numpy(dtype=float, na_value=np.nan),
                               rtol=0, atol=0, equal_nan=True).all()), (
            f'{source}: source field {c} unexpectedly modified'
        )

    # Evaluate recoding using the variable's SAV label semantics, NOT global
    # hard-coded meanings for numbers 8/9/98/99.
    recode_rows = []
    for col, expected in EXPECTED_RECODED.items():
        codes = missing_codes(meta, col)
        assert codes, f'{source}: no SAV-labelled nonresponses for {col}'
        original = num(raw[col])
        observed = num(clean[col])
        recoded = original.isin(codes)
        target = original.mask(recoded)
        assert bool(np.isclose(target.to_numpy(dtype=float, na_value=np.nan),
                               observed.to_numpy(dtype=float, na_value=np.nan),
                               rtol=0, atol=0, equal_nan=True).all()), (
            f'{source}: {col} changed values other than labelled nonresponses'
        )
        assert not observed.isin(codes).any(), f'{source}: uncleaned missing codes in {col}'
        n = int(recoded.sum())
        assert n == expected, f'{source}: {col} recoded {n}, expected {expected}'
        recode_rows.append({
            'source': source, 'variable': col,
            'label_inferred_missing_codes': ','.join(str(int(x)) for x in sorted(codes)),
            'recoded_to_nan': n, 'valid_after': int(observed.notna().sum()),
            'missing_after': int(observed.isna().sum()), 'passed': True,
        })

    # Legitimate categories that superficially resemble missing codes.
    for col in ('aireg_us', 'aireg_china', 'aireg_eu', 'aireg_country'):
        assert int(num(clean[col]).eq(8).sum()) == int(num(raw[col]).eq(8).sum())
    assert int(num(clean.internetfreq).eq(6).sum()) == 3303
    assert int(num(clean.age).eq(97).sum()) == 3

    bands = normalized_bands(clean.age_band)
    actual_bands = {name: int(bands.eq(name).sum()) for name in EXPECTED_BANDS}
    assert actual_bands == EXPECTED_BANDS, f'{source}: age-band totals differ: {actual_bands}'
    assert int(bands.isna().sum()) == 185, f'{source}: 185 missing age bands expected'
    assert int(clean.age_band_recovered.sum()) == 111
    assert int(clean.loc[num(raw.country).ne(13), 'age_band'].isna().sum()) == 183
    assert int(clean.loc[num(raw.country).eq(13), 'age_band'].isna().sum()) == 2

    gender = clean.gender_label.astype('string')
    actual_gender = {name: int(gender.eq(name).sum()) for name in ('Male', 'Female', 'Other')}
    actual_gender['Missing'] = int(gender.isna().sum())
    assert actual_gender == EXPECTED_GENDER, f'{source}: gender totals differ: {actual_gender}'

    # We retain original d_educ_* codes for P2. Report rather than falsely
    # claiming all 64 selected source columns have been recoded.
    source_rows = []
    for c in sorted(col for col in clean if col.startswith('d_educ_')):
        remaining = int(num(clean[c]).isin(missing_codes(meta, c)).sum())
        if remaining:
            source_rows.append({
                'source': source, 'retained_raw_column': c,
                'labelled_nonresponses_still_raw': remaining,
                'reason': 'Original education codes retained pending P2 educ3 mapping',
            })
    result = {
        'source': source, 'respondents': len(clean), 'part_i_source_columns': 64,
        'q30_recode_n': 808, 'q31_recode_n': 1964,
        'q32_refusals_across_items': 1536,
        'internet_nonresponse_recoded': 94,
        'internet_nonusers_preserved': 3303,
        'age_nonresponse_recoded': 294, 'age_bands_recovered': 111,
        'age_bands_known': int(bands.notna().sum()),
        'age_bands_missing_total': int(bands.isna().sum()),
        'age_bands_missing_outside_australia': 183,
        'age_bands_missing_australia': 2,
        'gender_other_retained': actual_gender['Other'],
        'gender_missing': actual_gender['Missing'],
        'retained_education_fields_with_labelled_nonresponses': len(source_rows),
        'passed': True,
    }
    print(f'PASS: {source}: 28,333 rows; 185 missing age bands; 111 recoveries; 21 Other genders kept')
    return result, recode_rows, source_rows


def self_test() -> None:
    """Prove the comparison detects both numeric and age-band corruption."""
    a = pd.DataFrame({'id': [1, 2], 'age_band': ['18–29', None],
                      'gender_label': ['Male', 'Other'], 'ai_heard': [1., np.nan]})
    b = pd.DataFrame({'id': [2, 1], 'age_band': [pd.NA, '18-29'],
                      'gender_label': ['Other', 'Male'], 'ai_heard': [np.nan, 1.]})
    compare_columns_by_id(a, b, ['age_band', 'gender_label', 'ai_heard'], stage='self-test identical')
    altered = b.copy()
    altered.loc[altered.id.eq(1), 'age_band'] = '30-49'
    try:
        compare_columns_by_id(a, altered, ['age_band'], stage='self-test altered age')
    except AssertionError:
        print('PASS: incorrectly assigned age band detected')
    else:
        raise AssertionError('Failed to catch changed age band')
    altered = b.copy()
    altered.loc[altered.id.eq(1), 'ai_heard'] = 8
    try:
        compare_columns_by_id(a, altered, ['ai_heard'], stage='self-test altered Q30')
    except AssertionError:
        print('PASS: uncleaned/corrupted Q30 response detected')
    else:
        raise AssertionError('Failed to catch corrupted Q30')
    fake_meta = SimpleNamespace(variable_value_labels={
        'ai_heard': {1: 'A lot', 8: "Don't know", 9: 'Refused'},
        'aireg_us': {1: 'A lot of trust', 8: 'Not sure', 9: 'Refused'},
    })
    assert missing_codes(fake_meta, 'ai_heard') == {8., 9.}
    assert missing_codes(fake_meta, 'aireg_us') == {9.}
    print('PASS: codebook semantics distinguish valid Q32 Not sure from Q30 nonresponse')
    print('TASK 18 SELF-TEST PASSED')


def main() -> None:
    import pyreadstat
    if not SAV.is_file() or not CSV.is_file():
        raise FileNotFoundError('Place both original SAV and CSV under data/raw/ before Task 18.')
    print('=== TASK 18: INTEGRATED POST-CLEANING ASSERTIONS ===')
    raw_sav, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    raw_csv = pd.read_csv(CSV, encoding='utf-8-sig', low_memory=False)
    raw_sav.columns = raw_sav.columns.str.strip().str.lower()
    raw_csv.columns = raw_csv.columns.str.strip().str.lower()
    assert raw_sav.shape == raw_csv.shape == (N, 483)
    previous_sav = raw_sav.copy(deep=True)
    previous_csv = raw_csv.copy(deep=True)

    # Run ACTUAL PRODUCTION cleaning. Neither run writes files.
    cleaned_sav = do_clean(raw_sav, meta, log=[])
    cleaned_csv = do_clean(raw_csv, meta, log=[])
    pd.testing.assert_frame_equal(raw_sav, previous_sav, check_exact=True)
    pd.testing.assert_frame_equal(raw_csv, previous_csv, check_exact=True)
    print('PASS: original SAV and CSV DataFrames unchanged by production cleaning')

    summary, recodes, retained = [], [], []
    for name, raw, clean in [('SAV', raw_sav, cleaned_sav), ('CSV', raw_csv, cleaned_csv)]:
        record, audit, carry = check_postclean(raw, clean, meta, source=name)
        summary.append(record)
        recodes += audit
        retained += carry

    compared = [*EXPECTED_RECODED.keys(), 'gender_clean', 'gender_label',
                'age_band', 'age_band_recovered', 'country', 'weight']
    compare_columns_by_id(cleaned_sav, cleaned_csv, compared, stage='production SAV vs CSV')

    # Independent implementation from Tasks 14, 16, 17, not the same methods
    # that built the src.cleaning results above.
    for source, raw, integrated in [('SAV', raw_sav, cleaned_sav), ('CSV', raw_csv, cleaned_csv)]:
        independent = independent_age_gender(raw, meta)
        a = integrated[['id', 'age', 'age_band', 'gender_clean', 'gender_label']].rename(
            columns={'age': 'age_clean'}
        )
        compare_columns_by_id(a, independent,
                              ['age_clean', 'age_band', 'gender_clean', 'gender_label'],
                              stage=f'{source} production vs independent Tasks 14/16/17')

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / 'task18_postclean_summary.csv', index=False)
    pd.DataFrame(recodes).to_csv(OUT / 'task18_nonresponse_audit.csv', index=False)
    pd.DataFrame(retained, columns=['source', 'retained_raw_column',
        'labelled_nonresponses_still_raw', 'reason']).to_csv(
            OUT / 'task18_retained_education_codes.csv', index=False)
    pd.DataFrame([{
        'task': 18,
        'decision': 'Assert cleaned analytical responses and unified age/gender, while retaining original country education variables for P2',
        'alternative': 'Assert all 64 original source columns have no nonresponse codes, or drop every respondent with any missing data',
        'reason': 'Retained education source codes are not yet harmonised; missing values and structural blanks must not cause row deletion',
        'effect': '28,333 rows retained; 28,148 known age bands; 185 missing overall (183 non-Australia, 2 Australia); no invalid code in cleaned outcomes',
        'verification': 'SAV+CSV; independent Tasks 14/16/17 calculations; label-derived recoding',
        'status': 'VERIFIED',
    }]).to_csv(OUT / 'task18_decision_log_entry.csv', index=False)
    pd.DataFrame([{
        'task': 18,
        'ai_contribution': 'Generated integrated post-clean assertions and cross-implementation validation code',
        'verification': 'Executed by student against SAV and CSV; compare to independently verified prior task scripts',
        'ai_error_or_limitation': 'Earlier 183 figure was non-Australian only; globally correct total is 185. Education fields are still raw.',
        'fix': 'Assert 183 outside Australia + 2 in Australia = 185; scope missing-code assertions to cleaned analytical columns',
        'status': 'VERIFIED',
    }]).to_csv(OUT / 'task18_ai_use_log_entry.csv', index=False)
    pd.DataFrame([{
        'task': 18, 'original_ai_or_checklist_claim': '183 respondents without age band across full data (without qualification)',
        'verified_result': '183 outside Australia; 2 in Australia; 185 in full 28,333-row dataset',
        'verification': 'Task16 original SAV/CSV per-country age audit and integrated Task18 assertions',
        'correction': 'Report both geography-specific and full-sample counts; do not force 183 as full-sample total',
    }]).to_csv(OUT / 'task18_ai_discrepancy_entry.csv', index=False)
    print('\nSAVED: task18_postclean_summary.csv, task18_nonresponse_audit.csv,')
    print('       task18_retained_education_codes.csv, Task18 decision/AI log entries')
    print(f'NOTE: {len([r for r in retained if r["source"] == "SAV"])} retained education source fields still contain labelled nonresponse codes; this is intentional until P2.')
    print('TASK 18 PASSED: combined SAV/CSV pipeline and independent age/gender checks agree; no source data changed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    options = parser.parse_args()
    if options.self_test:
        self_test()
    else:
        main()
