"""COMP5122M P1 — Task 7: audit observed response codes against the SAV codebook.

From the repository root:
    python scripts/validate_task07.py --self-test
    python scripts/validate_task07.py

We audit the ORIGINAL SAV (not the cleaned CSV). We never recode or delete data.
An unlabelled code is a data-quality finding, not automatic permission to alter it.

IMPORTANT: An SPSS *value label* is not required for every numeric or date
value. We make documented exceptions for direct-entry age, schooling-years,
household-size variables, respondent ID, weight and date fields. Special
response codes for those variables still have to agree with SAV value labels.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAV = ROOT / 'data' / 'raw' / 'Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = ROOT / 'data' / 'raw' / 'Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
OUT = ROOT / 'data' / 'clean'

# These exceptions describe how the QUESTIONS were recorded. They are not a
# list of values 'borrowed' from observed data to suppress errors.
# The first four variables have numeric, respondent-entered answers and labels
# only for exceptional special codes.
FREE_ENTRY = {
    'age': '18–96 years entered directly; code 97 means 97+ and is labelled',
    'd_educ_israel_years': 'integer years of education entered directly; 98/99 are labelled',
    'd_hhpeople': '1–96 entered directly; code 97 is labelled 97+',
    'd_adults': '1–7 entered directly; code 8 is labelled 8+',
}

# Legitimate fields with no value-label dictionary at all. Two are quantities
# and three contain dates. Inspect any OTHER field without value labels.
NO_VALUE_LABEL_EXPECTED = {
    'id': 'respondent identifier',
    'weight': 'continuous sampling weight',
    'qdate_s': 'survey start date',
    'qdate_e': 'survey end date',
    'intdate_aus': 'Australian interview date',
}


def canonical_code(value: object) -> str:
    """Compare numeric labels and observations, independent of float formatting."""
    if value is None or pd.isna(value):
        return ''
    s = str(value).strip()
    if not s:
        return ''
    try:
        x = float(s)
    except (ValueError, OverflowError):
        return s
    if not math.isfinite(x):
        return s
    return str(int(x)) if x.is_integer() else format(x, '.15g')


def is_valid_free_entry(var: str, code: str) -> bool:
    """Only verify meaningful numeric ranges established by question wording.

    For Israel's years of education, no explicit upper bound is imposed by
    the codebook; flag negatives/non-integers, not arbitrarily large years.
    """
    try:
        val = float(code)
    except (ValueError, OverflowError):
        return False
    if not math.isfinite(val) or not val.is_integer():
        return False
    num = int(val)
    if var == 'age':
        return 18 <= num <= 96
    if var == 'd_educ_israel_years':
        return num >= 0
    if var == 'd_hhpeople':
        return 1 <= num <= 96
    if var == 'd_adults':
        return 1 <= num <= 7
    return False


def audit(df: pd.DataFrame, raw_labels: dict, source: str = 'SAV') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one summary per variable and one finding per unlabelled code.

    Observe original codes directly; system-missing cells and CSV empty strings
    are not 'stray codes'.  Do not replace missing response codes with NaN.
    """
    df = df.rename(columns=lambda v: str(v).strip().lower())
    assert df.shape == (28_333, 483), f'{source}: unexpected dimensions {df.shape}'
    assert df.columns.is_unique, 'Duplicate columns after name normalisation'

    value_labels = {str(v).strip().lower(): {
        canonical_code(code): label for code, label in (labels or {}).items()
    } for v, labels in raw_labels.items()}
    assert set(value_labels).issubset(df.columns), 'SAV labels reference nonexistent columns'

    summary = []
    findings = []
    for var in df.columns:
        labels = value_labels.get(var, {})
        no_labels = not labels
        mode = ('no_value_labels_expected' if var in NO_VALUE_LABEL_EXPECTED
                else 'free_numeric_entry' if var in FREE_ENTRY
                else 'categorical_labelled' if labels
                else 'MISSING_CODEBOOK_DEFINITION')
        # Use value_counts on ORIGINAL values. Blank CSV strings are not answers.
        s = df[var]
        counts = s.value_counts(dropna=True)
        unlabelled_rows = 0
        free_entry_rows = 0
        observed_values = 0
        for raw_code, n in counts.items():
            code = canonical_code(raw_code)
            if not code:
                continue  # blank/whitespace in CSV denotes system missing
            n = int(n)
            observed_values += n
            if code in labels:
                continue
            if var in FREE_ENTRY and is_valid_free_entry(var, code):
                free_entry_rows += n
                continue
            if var in NO_VALUE_LABEL_EXPECTED:
                continue
            unlabelled_rows += n
            findings.append({
                'source': source,
                'variable': var,
                'observed_code': code,
                'n_affected': n,
                'issue': ('codebook has no labels for variable' if no_labels
                          else 'observed value not among codebook labels or documented direct-entry values'),
                'proposed_action': 'Investigate source documentation; preserve original value; do not guess a replacement',
                'example_ids': ', '.join(map(str, df.loc[s.map(canonical_code).eq(code), 'id'].head(5).tolist())),
            })
        summary.append({
            'source': source,
            'variable': var,
            'audit_type': mode,
            'n_codebook_labels': len(labels),
            'n_original_nonblank_values': observed_values,
            'n_system_missing_or_blank': len(s) - observed_values,
            'n_valid_unlabelled_direct_entries': free_entry_rows,
            'n_unlabelled_responses': unlabelled_rows,
            'n_unique_unlabelled_codes': sum(x['variable'] == var for x in findings),
            'note': FREE_ENTRY.get(var, NO_VALUE_LABEL_EXPECTED.get(var, '')),
        })

    summaries = pd.DataFrame(summary)
    flags = pd.DataFrame(findings, columns=[
        'source','variable','observed_code','n_affected',
        'issue','proposed_action','example_ids'])
    assert len(summaries) == 483 and summaries.variable.is_unique
    return summaries, flags


def self_test() -> None:
    # Test the classification functions, including the crucial special cases.
    assert canonical_code(98.0) == canonical_code('98') == '98'
    assert canonical_code('   ') == ''
    assert is_valid_free_entry('age', '45')
    assert not is_valid_free_entry('age', '98')
    assert is_valid_free_entry('d_educ_israel_years', '57')
    assert is_valid_free_entry('d_hhpeople', '6')
    assert not is_valid_free_entry('d_hhpeople', '-2')
    assert is_valid_free_entry('d_adults', '7')
    assert not is_valid_free_entry('d_adults', '9')
    print('PASS: continuous numeric entries are not mistaken for stray codes')
    print('PASS: unknown special response codes are not silently treated as valid')
    print('PASS: blank cells are not counted as observed codes')
    print('TASK 7 SELF-TEST PASSED')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not SAV.is_file():
        raise FileNotFoundError(f'Expected original SAV: {SAV}')
    import pyreadstat
    raw, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    labels = meta.variable_value_labels

    print('=== TASK 7: OBSERVED VALUES VS SAV CODEBOOK ===')
    summary, flags = audit(raw, labels)
    print('Original SAV:', raw.shape)
    print('Variables with value labels:', int(summary.n_codebook_labels.gt(0).sum()))
    print('Variables without value labels:', int(summary.n_codebook_labels.eq(0).sum()))
    print('Fields with legitimate direct-entry numbers:', len(FREE_ENTRY))
    print('Unlabelled codes needing review:', len(flags))
    print('Respondent answers affected:', int(flags.n_affected.sum()))
    if not flags.empty:
        print('\nExceptions requiring review:')
        print(flags[['variable', 'observed_code', 'n_affected', 'issue']].to_string(index=False))

    # Task 5 established row/column equality; independently cross-check the
    # same codebook audit against CSV to guard against data-loading differences.
    if CSV.is_file():
        csv = pd.read_csv(CSV, low_memory=False, encoding='utf-8-sig')
        csv_summary, csv_flags = audit(csv, labels, source='CSV')
        left = flags[['variable', 'observed_code', 'n_affected']].sort_values(
            ['variable', 'observed_code']).reset_index(drop=True)
        right = csv_flags[['variable', 'observed_code', 'n_affected']].sort_values(
            ['variable', 'observed_code']).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)
        print('\nPASS: SAV and CSV have identical unlabelled-code findings.')
    else:
        print('\nNOTE: CSV not found; independent CSV comparison skipped.')

    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / 'task07_codebook_audit.csv'
    flags_path = OUT / 'task07_unlabelled_codes.csv'
    summary.to_csv(summary_path, index=False)
    flags.to_csv(flags_path, index=False)
    print('Saved:', summary_path)
    print('Saved:', flags_path)
    print('\nReview the flagged rows and record your decision. Never invent a label.')

    # This is an *independent expected finding*, not an allowed response code.
    # It does not suppress or repair the anomaly.
    expected = [('d_income2_france', '98', 5)]
    actual = sorted(zip(flags.variable, flags.observed_code, flags.n_affected))
    if actual == expected:
        print('\nTASK 7 AUDIT COMPLETE: one genuine exception identified and logged.')
        print('NOTE: "No stray codes" is NOT literally true: French income follow-up code 98 appears five times.')
        print('All other coded variables conform to their labels/direct-entry definitions.')
    else:
        print('\nTASK 7 REVIEW REQUIRED: the SAV findings differ from the independently checked dictionary/CSV reference.')
        raise AssertionError(f'Expected {expected}; found {actual}. Investigate and log the discrepancy.')


if __name__ == '__main__':
    main()
