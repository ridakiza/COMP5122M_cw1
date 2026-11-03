"""COMP5122M P1 — Task 12: clean internet-frequency nonresponses.

Run from the repository root:
    python scripts/validate_task12.py --self-test
    python scripts/validate_task12.py

Recode only the nonresponse codes inferred from the original SAV value labels.
Preserve code 6 ('does not use the internet'). Treat the 1,829 Australian
structural blanks separately from the 94 recorded nonresponses elsewhere.

Uses a new DataFrame; original Pew files and pew_clean.csv are not modified.
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
COLS = ['id', 'country', 'internetfreq']
VAR = 'internetfreq'
EXPECTED_ROWS = 28_333
EXPECTED_COUNTRIES = 24
VALID_CODES = {1, 2, 3, 4, 5, 6, 8, 9}

# Independently counted from the uploaded, ORIGINAL CSV. Verification only:
# these values NEVER determine how to clean a response.
REFERENCE_COUNTS = {1: 7861, 2: 11152, 3: 2182, 4: 986,
                    5: 926, 6: 3303, 8: 57, 9: 37}
REFERENCE_STRUCTURAL_BLANKS = 1829


def sav_labels(meta: object) -> dict[int, str]:
    """Extract labels only from original SAV metadata (not numeric assumptions)."""
    mappings = getattr(meta, 'variable_value_labels', {}) or {}
    keys = [k for k in mappings if str(k).strip().casefold() == VAR]
    assert len(keys) == 1, f'Expected one SAV label mapping for {VAR}, found {keys}'
    labels = {}
    for raw_code, label in mappings[keys[0]].items():
        as_float = float(raw_code)
        assert as_float.is_integer(), f'Non-integer code in metadata: {raw_code!r}'
        code = int(as_float)
        assert code not in labels, f'Duplicate code in SAV metadata: {code}'
        labels[code] = str(label)
    assert set(labels) == VALID_CODES, f'Unexpected codes in SAV labels: {labels}'
    return labels


def codes_to_recode(meta: object) -> set[int]:
    labels = sav_labels(meta)
    inferred = {code for code, label in labels.items() if is_nonresponse(label)}
    # Safety checks on *meaning*, not rules for replacing values.
    text6 = canonical_label(labels[6])
    assert 'internet' in text6 and ('not use' in text6 or 'never' in text6), (
        f'Code 6 label is unexpected: {labels[6]!r}'
    )
    assert 6 not in inferred, 'Valid non-internet-user response was classified as missing'
    assert "don't know" in canonical_label(labels[8]) or 'do not know' in canonical_label(labels[8]), (
        f'Unexpected label for code 8: {labels[8]!r}'
    )
    assert 'refus' in canonical_label(labels[9]), f'Unexpected code 9 label: {labels[9]!r}'
    assert inferred == {8, 9}, f'Inferred nonresponse codes differ from questionnaire: {inferred}'
    return inferred


def normalize_frame(df: pd.DataFrame, source: str, *, expected: bool = True) -> pd.DataFrame:
    """Numeric comparison copy. Strip whitespace-only CSV cells to real blanks."""
    data = df.copy(deep=True)
    data.columns = data.columns.str.strip().str.lower()
    assert set(COLS).issubset(data.columns), f'{source}: missing columns {set(COLS) - set(data.columns)}'
    data = data[COLS].copy(deep=True)
    for col in COLS:
        if pd.api.types.is_string_dtype(data[col]):
            stripped = data[col].astype('string').str.strip()
            data[col] = stripped.mask(stripped.eq(''), pd.NA)
        # Same dtype for SAV (float64) and CSV (Int64 from strings), so IDs compare equal.
        data[col] = pd.to_numeric(data[col], errors='raise').astype('float64')
    assert data.id.notna().all() and data.id.is_unique, f'{source}: respondent IDs not unique or missing'
    assert data.country.notna().all(), f'{source}: country blank'
    assert data.country.eq(np.floor(data.country)).all(), f'{source}: non-integer country code'
    observed = data[VAR].dropna()
    assert np.isfinite(observed.to_numpy(dtype=float)).all(), f'{source}: nonfinite response'
    assert set(observed.unique()).issubset(VALID_CODES), (
        f'{source}: unrecognised internet-frequency codes {set(observed.unique()) - VALID_CODES}'
    )
    if expected:
        assert len(data) == EXPECTED_ROWS, f'{source}: unexpected row count {len(data)}'
        assert data.country.nunique() == EXPECTED_COUNTRIES, f'{source}: wrong country count'
    return data


def original_counts(df: pd.DataFrame) -> dict[int, int]:
    return {code: int(df[VAR].eq(code).sum()) for code in sorted(VALID_CODES)}


def verify_routing(df: pd.DataFrame, meta: object, *, expected: bool = True) -> tuple[int, int]:
    """Only Australia may have structural blanks; others have response codes."""
    labels = getattr(meta, 'variable_value_labels', {}) or {}
    country_keys = [k for k in labels if str(k).strip().casefold() == 'country']
    assert len(country_keys) == 1, 'Missing country labels from original SAV'
    country_map = labels[country_keys[0]]
    australia_codes = [int(float(k)) for k, v in country_map.items()
                       if str(v).strip().casefold() == 'australia']
    assert len(australia_codes) == 1, f'Expected one Australia code: {australia_codes}'
    aus_mask = df.country.eq(australia_codes[0])
    outside_blank = int(df.loc[~aus_mask, VAR].isna().sum())
    australia_present = int(df.loc[aus_mask, VAR].notna().sum())
    assert outside_blank == 0, f'{outside_blank} blanks outside Australia need review'
    assert australia_present == 0, f'{australia_present} Australia responses unexpectedly recorded'
    n_aus = int(aus_mask.sum())
    if expected:
        assert n_aus == REFERENCE_STRUCTURAL_BLANKS, f'Australia count {n_aus} does not match reference'
    return n_aus, outside_blank


def clean_internet(df: pd.DataFrame, meta: object, *, reference: bool = True,
                   show: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean copy, one-row audit). Does not mutate the supplied frame."""
    before = df.copy(deep=True)
    labels = sav_labels(meta)
    bad_codes = codes_to_recode(meta)
    counts = original_counts(df)
    australia_blanks, _ = verify_routing(df, meta, expected=reference)
    if reference:
        assert counts == REFERENCE_COUNTS, f'Code counts do not match independent CSV reference: {counts}'
    assert sum(counts.values()) + australia_blanks == len(df), 'Unexpected missing responses'

    clean = df.copy(deep=True)
    mask = df[VAR].isin(bad_codes)
    clean[VAR] = df[VAR].mask(mask, np.nan)

    n_converted = int(mask.sum())
    n_valid = int(clean[VAR].notna().sum())
    n_total_missing = int(clean[VAR].isna().sum())
    assert n_converted == counts[8] + counts[9]
    assert n_valid == len(df) - australia_blanks - n_converted
    assert n_total_missing == australia_blanks + n_converted
    assert not clean[VAR].isin(bad_codes).any()
    for code in (1, 2, 3, 4, 5, 6):
        assert clean.loc[df[VAR].eq(code), VAR].eq(code).all(), f'Valid code {code} was changed'
    assert int(clean[VAR].eq(6).sum()) == counts[6], 'Non-users were lost'
    assert clean.loc[df[VAR].isna(), VAR].isna().all(), 'Structural blanks changed'

    pd.testing.assert_frame_equal(df, before, check_exact=True)
    pd.testing.assert_frame_equal(df[['id', 'country']], clean[['id', 'country']], check_exact=True)
    assert clean is not df and len(clean) == len(df) and clean.index.equals(df.index)

    audit = pd.DataFrame([{
        'variable': VAR,
        'nonresponse_codes_from_sav_labels': ','.join(map(str, sorted(bad_codes))),
        **{f'code_{code}_original': counts[code] for code in sorted(VALID_CODES)},
        'australia_structural_blanks_preserved': australia_blanks,
        'original_recorded_outside_australia': len(df) - australia_blanks,
        'dont_know_to_nan': counts[8],
        'refused_to_nan': counts[9],
        'total_recoded_to_nan': n_converted,
        'valid_noninternetusers_kept': counts[6],
        'valid_after': n_valid,
        'missing_after_including_australia': n_total_missing,
        'respondents_removed': 0,
    }])
    if show:
        print('SAV labels:', labels)
        print('Missing codes inferred FROM labels:', sorted(bad_codes))
        print('Original counts:', counts)
        print(f'Before: {len(df) - australia_blanks:,} recorded; {australia_blanks:,} Australia structural blanks')
        print(f'Valid code 6 (non-users) preserved: {counts[6]:,}')
        print(f'Recoded: code 8 = {counts[8]}, code 9 = {counts[9]} ({n_converted} total)')
        print(f'After: {n_valid:,} valid; {n_total_missing:,} total missing '
              f'({australia_blanks:,} structural + {n_converted} recoded)')
        print('PASS: raw data unchanged; 28,333 rows preserved' if reference else 'PASS: original unchanged')
    return clean, audit


def compare_by_id(first: pd.DataFrame, second: pd.DataFrame, stage: str) -> None:
    a = first.set_index('id').sort_index()
    b = second.set_index('id').sort_index()
    assert a.index.equals(b.index), f'{stage}: SAV and CSV IDs differ'
    for column in ('country', VAR):
        left = a[column].to_numpy(dtype=float)
        right = b[column].to_numpy(dtype=float)
        matches = np.isclose(left, right, atol=0, rtol=0, equal_nan=True)
        assert matches.all(), f'{stage}: {column} differs for {(~matches).sum()} IDs'
    print(f'PASS: SAV/CSV agree for all {len(a):,} respondents {stage}')


def country_audit(original: pd.DataFrame, clean: pd.DataFrame, meta: object) -> pd.DataFrame:
    labels = getattr(meta, 'variable_value_labels', {}) or {}
    k = next(k for k in labels if str(k).strip().casefold() == 'country')
    names = {int(float(code)): str(label) for code, label in labels[k].items()}
    d = pd.DataFrame({
        'country_code': original.country.astype(int),
        'original_blank': original[VAR].isna(),
        'code_6_kept': original[VAR].eq(6),
        'code_8_to_nan': original[VAR].eq(8),
        'code_9_to_nan': original[VAR].eq(9),
        'valid_after': clean[VAR].notna(),
    })
    out = d.groupby('country_code', sort=True).agg(
        respondents=('country_code', 'size'),
        original_blanks=('original_blank', 'sum'),
        non_users_preserved=('code_6_kept', 'sum'),
        dont_know_recoded=('code_8_to_nan', 'sum'),
        refused_recoded=('code_9_to_nan', 'sum'),
        valid_after=('valid_after', 'sum'),
    ).reset_index()
    out.insert(1, 'country', out.country_code.map(names))
    assert len(out) == EXPECTED_COUNTRIES
    return out


def self_test() -> None:
    labels = {1: 'Almost constantly', 2: 'Several times a day', 3: 'About once a day',
              4: 'Several times a week', 5: 'Less often',
              6: 'Or do you not use the internet', 8: "Don't know (DO NOT READ)",
              9: 'Refused (DO NOT READ)'}
    meta = SimpleNamespace(variable_value_labels={
        VAR: labels,
        'country': {12: 'UK', 13: 'Australia'},
    })
    fake = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'country': [12] * 8 + [13],
        VAR: [1, 2, 3, 4, 5, 6, 8, 9, None],
    })
    x = normalize_frame(fake, 'self-test', expected=False)
    prior = x.copy(deep=True)
    cleaned, record = clean_internet(x, meta, reference=False, show=False)
    pd.testing.assert_frame_equal(x, prior, check_exact=True)
    assert int(record.total_recoded_to_nan.iloc[0]) == 2
    assert cleaned.loc[5, VAR] == 6 and int(cleaned[VAR].eq(6).sum()) == 1
    assert cleaned.loc[[6, 7, 8], VAR].isna().all()
    print('PASS: valid code 6 retained; only 8/9 recoded; Australia blank preserved')

    wrong = SimpleNamespace(variable_value_labels={**meta.variable_value_labels,
        VAR: {**labels, 6: "Don't know"}})
    try:
        clean_internet(x, wrong, reference=False, show=False)
    except AssertionError:
        print('PASS: incorrect label for code 6 causes validation failure')
    else:
        raise AssertionError('Failed to detect damaged code 6 label')

    wrong_values = fake.copy(deep=True)
    wrong_values.loc[0, VAR] = 7
    try:
        normalize_frame(wrong_values, 'self-test', expected=False)
    except AssertionError:
        print('PASS: unlabelled code 7 causes validation failure')
    else:
        raise AssertionError('Failed to detect unexpected response code 7')

    bad_csv = x.copy(deep=True)
    bad_csv.loc[5, VAR] = 5
    try:
        compare_by_id(x, bad_csv, 'self-test')
    except AssertionError:
        print('PASS: respondent-level SAV/CSV mismatch detected')
    else:
        raise AssertionError('Failed to detect respondent-level difference')

    print('TASK 12 SELF-TEST PASSED')


def main() -> None:
    print('=== TASK 12: INTERNET FREQUENCY — KEEP 6, RECODE 8/9 ===')
    for path in (SAV, CSV):
        if not path.is_file():
            raise FileNotFoundError(f'Missing original file: {path}')
    try:
        import pyreadstat
    except ImportError as exc:
        raise ImportError('Install pyreadstat: python -m pip install pyreadstat') from exc

    sav_raw, meta = pyreadstat.read_sav(str(SAV), usecols=COLS,
                                      apply_value_formats=False, user_missing=True)
    csv_raw = pd.read_csv(CSV, usecols=COLS, dtype='string',
                          encoding='utf-8-sig', low_memory=False)
    sav = normalize_frame(sav_raw, 'SAV')
    csv = normalize_frame(csv_raw, 'CSV')
    verify_routing(sav, meta)
    verify_routing(csv, meta)
    compare_by_id(sav, csv, 'before cleaning')

    clean_sav, audit_sav = clean_internet(sav, meta)
    clean_csv, audit_csv = clean_internet(csv, meta, show=False)
    pd.testing.assert_frame_equal(audit_sav, audit_csv, check_exact=True)
    compare_by_id(clean_sav, clean_csv, 'after cleaning')
    assert len(clean_sav) == len(sav) == EXPECTED_ROWS

    OUT.mkdir(parents=True, exist_ok=True)
    paths = [OUT / 'task12_internetfreq_audit.csv',
             OUT / 'task12_internetfreq_country_audit.csv',
             OUT / 'task12_validation_note.txt']
    audit_sav.to_csv(paths[0], index=False)
    country_audit(sav, clean_sav, meta).to_csv(paths[1], index=False)
    paths[2].write_text(
        'TASK 12 — internetfreq missing-response cleaning\n'
        'DECISION: Read each response label from original SAV metadata. '
        "Recode 'Don't know' (8, n=57) and 'Refused' (9, n=37) to NaN. "
        'Preserve code 6 (does not use the internet, n=3,303) as a valid response. '
        'Keep 1,829 Australian structurally blank responses separate from nonresponses.\n'
        'ALTERNATIVE: Remove code 6 as if it were a missing value (reject: 3,303 '
        'genuine nonusers would be lost); or drop Australian blank rows '
        '(reject: question was not administered in Australia).\n'
        'EFFECT: 94 recorded nonresponses converted to NaN, 0 rows removed, '
        '26,410 valid internet-frequency values remain, 1,923 total missing '
        '(1,829 structurally blank + 94 recoded). SAV and CSV compared by ID.\n'
        'AI USE: ChatGPT drafted the code and independently verified counts '
        'from uploaded original CSV; synthetic self-tests also provided. '
        'Human executed the authoritative SAV+CSV run separately.\n'
        'AI ERROR: Record any mismatch, faulty assertion, or required code correction '
        'in the project AI discrepancy log; none should be claimed without evidence.\n'
        'NOTE: Standalone validation does not overwrite data/clean/pew_clean.csv; '
        'integrate verified function in src/cleaning.py later.\n',
        encoding='utf-8',
    )
    print('\nSaved:', *(p.name for p in paths))
    print('TASK 12 PASSED: code 6 retained; 94 nonresponses recoded; '
          '1,829 Australian blanks preserved; SAV and CSV agree.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()