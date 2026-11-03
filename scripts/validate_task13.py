"""COMP5122M P1 — Task 13: clean exact ages without losing valid 97+ answers.

Run from repository root:
    python scripts/validate_task13.py --self-test
    python scripts/validate_task13.py

- Reads the original SAV with the original numeric codes and their SPSS labels.
- Infers missing answers from the meaning of the SAV value labels (Task 4).
- Converts age codes 98 and 99 to NaN on a COPY; keeps 97 meaning "97 or older".
- Checks Australian structural blanks separately; leaves age_australia untouched.
- Compares the SAV and CSV person-by-person before and after recoding.
- Prints a before/after audit; does NOT change the original files or pew_clean.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from validate_task04 import canonical_label, is_nonresponse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'clean'
SAV = RAW / 'Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = RAW / 'Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
COLS = ['id', 'country', 'age', 'age_australia']
N_ROWS = 28_333
N_COUNTRIES = 24

# Independent counts from the original uploaded CSV, for VALIDATION only.
# These do not determine which codes get recoded.
REFERENCE = {
    'australia_n': 1829,
    'age_australia_present': 1827,
    'age_97': 3,
    'age_98': 9,
    'age_99': 285,
    'age_recorded_before': 26504,
    'age_valid_after': 26210,
    'age_missing_after': 2123,  # 1829 structural + 294 recorded nonresponses
}


def value_labels(metadata: object, name: str) -> dict[int, str]:
    """Get original SAV labels by field name, keeping numeric codes as integers."""
    mappings = getattr(metadata, 'variable_value_labels', {}) or {}
    matches = [k for k in mappings if str(k).strip().casefold() == name]
    assert len(matches) == 1, f'Missing or ambiguous SAV metadata for {name!r}: {matches}'
    result = {}
    for raw_code, text in mappings[matches[0]].items():
        number = float(raw_code)
        assert number.is_integer(), f'Non-integer SAV label code: {raw_code!r}'
        code = int(number)
        assert code not in result, f'Duplicate SAV label code: {code}'
        result[code] = str(text)
    return result


def age_missing_codes(metadata: object) -> set[int]:
    """Infer recodes by *meaning*, then assert the questionnaire mapping."""
    labels = value_labels(metadata, 'age')
    assert set(labels) == {97, 98, 99}, f'Unexpected age special labels: {labels}'
    missing = {code for code, label in labels.items() if is_nonresponse(label)}
    assert '97' in canonical_label(labels[97]) and 'older' in canonical_label(labels[97]), (
        f'Code 97 must mean 97 or older, got {labels[97]!r}'
    )
    assert not is_nonresponse(labels[97]), 'Age 97+ was wrongly classified as a nonresponse'
    assert "don't know" in canonical_label(labels[98]) or 'do not know' in canonical_label(labels[98]), (
        f'Unexpected label for age code 98: {labels[98]!r}'
    )
    assert 'refus' in canonical_label(labels[99]), f'Unexpected label for age code 99: {labels[99]!r}'
    assert missing == {98, 99}, f'SAV missing codes do not match the questionnaire: {missing}'
    return missing


def normalise_frame(raw: pd.DataFrame, name: str, *, reference: bool = True) -> pd.DataFrame:
    """Make a numeric comparison copy; whitespace in CSV becomes blank only in this copy."""
    raw_columns = [str(c).strip().casefold() for c in raw.columns]
    assert len(set(raw_columns)) == len(raw_columns), f'{name}: duplicate column names'
    assert set(COLS).issubset(raw_columns), f'{name}: fields missing: {set(COLS) - set(raw_columns)}'
    renamed = dict(zip(raw.columns, raw_columns))
    working = raw.rename(columns=renamed)[COLS].copy(deep=True)
    for col in COLS:
        if pd.api.types.is_object_dtype(working[col]) or pd.api.types.is_string_dtype(working[col]):
            txt = working[col].astype('string').str.strip()
            working[col] = txt.mask(txt.eq(''), pd.NA)
        working[col] = pd.to_numeric(working[col], errors='raise')
        nonmissing = working[col].dropna().to_numpy(dtype=float)
        assert np.isfinite(nonmissing).all(), f'{name}: nonfinite values in {col}'
    assert working.id.notna().all() and working.id.is_unique, f'{name}: missing/duplicate IDs'
    assert working.country.notna().all(), f'{name}: missing country values'
    if reference:
        assert len(working) == N_ROWS, f'{name}: unexpected number of rows'
        assert working.country.nunique() == N_COUNTRIES, f'{name}: unexpected country count'
    return working


def australia_code(metadata: object) -> int:
    labels = value_labels(metadata, 'country')
    codes = [code for code, text in labels.items() if canonical_label(text) == 'australia']
    assert len(codes) == 1, f'Could not identify Australia from original SAV: {codes}'
    return codes[0]


def validate_routing(df: pd.DataFrame, metadata: object, *, reference: bool = True) -> int:
    aus = df.country.eq(australia_code(metadata))
    n_aus = int(aus.sum())
    assert df.loc[aus, 'age'].isna().all(), 'age must be blank in Australia'
    assert df.loc[~aus, 'age'].notna().all(), 'Age unexpectedly blank outside Australia'
    assert df.loc[~aus, 'age_australia'].isna().all(), 'age_australia must be blank outside Australia'
    if reference:
        assert n_aus == REFERENCE['australia_n'], f'Unexpected Australian sample size: {n_aus}'
        assert int(df.loc[aus, 'age_australia'].notna().sum()) == REFERENCE['age_australia_present']
    return n_aus


def clean_exact_age(df: pd.DataFrame, metadata: object, *, reference: bool = True,
                    show: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean exact age in a copy, returning (cleaned copy, audit DataFrame)."""
    original = df.copy(deep=True)
    missing_codes = age_missing_codes(metadata)
    n_aus = validate_routing(df, metadata, reference=reference)

    # Ordinary ages 18-96 are direct-entry numbers, so not individually SAV-labelled.
    # Code 97 is a valid top-coded age, NOT an exact age of 97.
    recorded = df['age'].dropna()
    assert recorded.between(18, 99).all(), 'Age contains a value outside 18..99'
    assert (recorded % 1 == 0).all(), 'Age contains fractional values'
    n_97 = int(df.age.eq(97).sum())
    n_98 = int(df.age.eq(98).sum())
    n_99 = int(df.age.eq(99).sum())
    n_before = int(df.age.notna().sum())
    original_blanks = int(df.age.isna().sum())

    cleaned = df.copy(deep=True)
    recode_mask = cleaned.age.isin(missing_codes)
    cleaned['age'] = cleaned['age'].mask(recode_mask, np.nan)
    n_recoded = int(recode_mask.sum())
    n_after = int(cleaned.age.notna().sum())
    n_missing_after = int(cleaned.age.isna().sum())

    assert n_recoded == n_98 + n_99
    assert n_before == n_after + n_recoded
    assert n_missing_after == original_blanks + n_recoded
    assert n_97 == int(cleaned.age.eq(97).sum()), 'Age 97+ responses were lost'
    assert cleaned.age.dropna().between(18, 97).all(), 'Unexpected cleaned ages'
    assert not cleaned.age.isin(missing_codes).any()
    assert cleaned.loc[df.age.isna(), 'age'].isna().all(), 'Structural blanks were changed'
    valid_mask = df.age.notna() & ~recode_mask
    pd.testing.assert_series_equal(
        cleaned.loc[valid_mask, 'age'], df.loc[valid_mask, 'age'], check_exact=True)
    pd.testing.assert_frame_equal(cleaned[['id', 'country', 'age_australia']],
                                  df[['id', 'country', 'age_australia']], check_exact=True)
    pd.testing.assert_frame_equal(df, original, check_exact=True)
    assert len(cleaned) == len(df) and cleaned.index.equals(df.index) and cleaned is not df

    if reference:
        for field, observed in [
            ('age_97', n_97), ('age_98', n_98), ('age_99', n_99),
            ('age_recorded_before', n_before), ('age_valid_after', n_after),
            ('age_missing_after', n_missing_after),
        ]:
            assert observed == REFERENCE[field], (
                f'{field}: SAV-derived {observed} differs from independent CSV reference {REFERENCE[field]}'
            )
    summary = pd.DataFrame([{
        'variable': 'age',
        'nonresponse_codes_from_sav_labels': ','.join(map(str, sorted(missing_codes))),
        'age_97_or_older_kept': n_97,
        'age_98_dont_know_to_nan': n_98,
        'age_99_refused_to_nan': n_99,
        'total_recoded_to_nan': n_recoded,
        'original_recorded': n_before,
        'australia_structural_blanks': original_blanks,
        'age_valid_after': n_after,
        'age_missing_after_including_australia': n_missing_after,
        'age_australia_present_unchanged': int(df.age_australia.notna().sum()),
        'respondents_removed': 0,
    }])
    if show:
        print('SAV age special-value labels:', value_labels(metadata, 'age'))
        print('Missing response codes derived FROM labels:', sorted(missing_codes))
        print(f'Before: {n_before:,} recorded ages; {original_blanks:,} Australian structural blanks')
        print(f'Keep: code 97 (97 or older): {n_97} people')
        print(f'Recode: code 98 (Don\'t know) = {n_98}; code 99 (Refused) = {n_99}; total = {n_recoded}')
        print(f'After: {n_after:,} usable age values; {n_missing_after:,} missing in age '
              f'({n_aus:,} Australian structural + {n_recoded} recoded)')
        print(f'Australian age-band answers left untouched: {int(df.age_australia.notna().sum()):,}')
        print('PASS: original DataFrame unchanged; 28,333 rows and all other values preserved'
              if reference else 'PASS: original unchanged and all valid values preserved')
    return cleaned, summary


def compare_by_id(a: pd.DataFrame, b: pd.DataFrame, stage: str) -> None:
    a = a.set_index('id').sort_index()
    b = b.set_index('id').sort_index()
    assert a.index.equals(b.index), f'{stage}: SAV/CSV respondent IDs do not match'
    for col in ('country', 'age', 'age_australia'):
        left = a[col].to_numpy(dtype=float)
        right = b[col].to_numpy(dtype=float)
        equal = np.isclose(left, right, atol=0, rtol=0, equal_nan=True)
        assert equal.all(), f'{stage}: {int((~equal).sum())} SAV/CSV mismatches in {col}'
    print(f'PASS: SAV/CSV agree for all {len(a):,} respondents {stage}')


def country_audit(original: pd.DataFrame, cleaned: pd.DataFrame, meta: object) -> pd.DataFrame:
    labels = value_labels(meta, 'country')
    t = pd.DataFrame({
        'country_code': original.country.astype(int),
        'code_97_kept': original.age.eq(97),
        'code_98_recoded': original.age.eq(98),
        'code_99_recoded': original.age.eq(99),
        'original_age_blank': original.age.isna(),
        'valid_age_after': cleaned.age.notna(),
        'australia_age_band_present': original.age_australia.notna(),
    })
    result = t.groupby('country_code', sort=True).agg(
        n_respondents=('valid_age_after', 'size'),
        code_97_kept=('code_97_kept', 'sum'),
        code_98_recoded=('code_98_recoded', 'sum'),
        code_99_recoded=('code_99_recoded', 'sum'),
        original_age_blank=('original_age_blank', 'sum'),
        valid_age_after=('valid_age_after', 'sum'),
        australia_age_band_present=('australia_age_band_present', 'sum'),
    ).reset_index()
    result.insert(1, 'country', result.country_code.map(labels))
    result['missing_age_after'] = result.n_respondents - result.valid_age_after
    assert len(result) == N_COUNTRIES
    return result


def self_test() -> None:
    meta = SimpleNamespace(variable_value_labels={
        'age': {97: '97 or older', 98: "Don't know", 99: 'Refused'},
        'country': {2: 'Canada', 13: 'Australia'},
    })
    toy = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7],
        'country': [2, 2, 2, 2, 2, 13, 13],
        'age': [18, 65, 97, 98, 99, np.nan, np.nan],
        'age_australia': [np.nan, np.nan, np.nan, np.nan, np.nan, 2, np.nan],
    })
    old = toy.copy(deep=True)
    clean, audit = clean_exact_age(toy, meta, reference=False, show=False)
    assert clean.age.tolist()[:3] == [18, 65, 97]
    assert clean.age.isna().sum() == 4  # two recorded nonresponses + two Australian structural blanks
    assert audit.loc[0, 'total_recoded_to_nan'] == 2
    assert clean.age_australia.equals(toy.age_australia)
    pd.testing.assert_frame_equal(toy, old, check_exact=True)
    print('PASS: 98/99 recoded, valid 97+ kept, and Australian blanks preserved')

    wrong_meta = SimpleNamespace(variable_value_labels={
        'age': {97: 'Refused', 98: "Don't know", 99: 'Refused'},
        'country': {2: 'Canada', 13: 'Australia'},
    })
    try:
        clean_exact_age(toy, wrong_meta, reference=False, show=False)
    except AssertionError:
        print('PASS: incorrect SAV value label for 97 rejected')
    else:
        raise AssertionError('Incorrect code 97 label was not detected')

    bad_values = toy.copy(deep=True)
    bad_values.loc[0, 'age'] = 105
    try:
        clean_exact_age(bad_values, meta, reference=False, show=False)
    except AssertionError:
        print('PASS: unknown or impossible numeric age values rejected')
    else:
        raise AssertionError('Invalid age value went undetected')

    bad_csv = toy.copy(deep=True)
    bad_csv.loc[1, 'age'] = 66
    try:
        compare_by_id(toy, bad_csv, 'synthetic mismatch test')
    except AssertionError:
        print('PASS: deliberate SAV/CSV age mismatch detected')
    else:
        raise AssertionError('SAV/CSV mismatch went undetected')
    print('TASK 13 SELF-TEST PASSED')


def main() -> None:
    import pyreadstat
    for path in (SAV, CSV):
        if not path.is_file():
            raise FileNotFoundError(f'Missing original input: {path}')
    print('=== TASK 13: CLEAN AGE (KEEP 97+, RECODE 98/99) ===')
    sav_raw, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    csv_raw = pd.read_csv(CSV, low_memory=False, encoding='utf-8-sig')
    assert sav_raw.shape == (N_ROWS, 483) and csv_raw.shape == (N_ROWS, 483), 'Unexpected file dimensions'
    sav = normalise_frame(sav_raw, 'SAV')
    csv = normalise_frame(csv_raw, 'CSV')
    compare_by_id(sav, csv, 'before cleaning')

    sav_clean, audit = clean_exact_age(sav, meta)
    csv_clean, _ = clean_exact_age(csv, meta, show=False)
    compare_by_id(sav_clean, csv_clean, 'after cleaning')

    OUT.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT / 'task13_age_audit.csv', index=False)
    country_audit(sav, sav_clean, meta).to_csv(OUT / 'task13_age_country_audit.csv', index=False)
    note = (
        'Task 13 — exact-age nonresponse cleaning\n'
        'Decision: use original SAV response labels to identify nonresponses; '
        'convert only age 98 (Don\'t know) and 99 (Refused) to NaN.\n'
        'Alternative rejected: recode 97 as missing, or replace 97 with guessed exact ages.\n'
        'Reason: code 97 means 97 or older; retain as an upper-coded value.\n'
        f'Effect: {int(audit.loc[0, "total_recoded_to_nan"])} responses recoded, '
        f'{int(audit.loc[0, "age_97_or_older_kept"])} top-coded ages preserved, '
        '0 respondents removed; Australia remains structural blank in the age column.\n'
        'Task 14 will recover only defensible AGE BANDS from the country-specific follow-ups; '
        'this task does not impute ages.\n'
        'AI use: AI provided and tested the validation script against independent CSV counts; '
        'user checks the original SAV/CSV execution. Record any detected error with correction.\n'
    )
    (OUT / 'task13_validation_note.txt').write_text(note, encoding='utf-8')
    # Separate, rerunnable CSV entries: import these into the team's cumulative logs.
    pd.DataFrame([{
        'task': 13,
        'decision': 'Recode only SAV-labelled age nonresponses 98 and 99',
        'alternative': 'Treat 97 as missing or infer exact ages from follow-up bands',
        'reason': '97 is a legitimate top-coded age (97 or older); follow-ups do not give exact ages',
        'responses_recoded': int(audit.loc[0, 'total_recoded_to_nan']),
        'valid_97_plus_preserved': int(audit.loc[0, 'age_97_or_older_kept']),
        'respondents_removed': 0,
        'verification': 'SAV value labels + CSV reference counts + person-by-person SAV/CSV comparison',
        'status': 'passed',
    }]).to_csv(OUT / 'task13_decision_log_entry.csv', index=False)
    pd.DataFrame([{
        'task': 13,
        'ai_contribution': 'Suggested and wrote age recoding, audit and validation code',
        'independent_evidence': 'Original uploaded CSV + Pew age response labels',
        'human_verification': 'Original SAV run and SAV/CSV comparison after cleaning',
        'verification_outcome': 'passed',
        'ai_errors_identified': 'none during this task verification',
        'correction': '',
    }]).to_csv(OUT / 'task13_ai_use_log_entry.csv', index=False)
    print('Saved: task13_age_audit.csv task13_age_country_audit.csv task13_validation_note.txt')
    print('       task13_decision_log_entry.csv task13_ai_use_log_entry.csv')
    print('TASK 13 PASSED: only age 98/99 recoded; valid 97+ and all Australian bands preserved; SAV/CSV agree.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true', help='Run toy-data checks without Pew files')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
