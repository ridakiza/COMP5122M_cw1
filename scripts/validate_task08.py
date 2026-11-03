"""COMP5122M P1 — Task 8: verify question routing by country.

Run from the repository root:
    python scripts/validate_task08.py --self-test
    python scripts/validate_task08.py

Read ORIGINAL SAV and CSV. Neither file is modified. Survey response codes like
8 / 9 are RECORDED answers here, not system blanks; whitespace-only fields are
structurally blank. We check that AI questions cover all 24 countries, that
Australia-specific variables contain no responses outside Australia, and that
the general age, gender and internet fields have the intended country routing.

Outputs in data/clean/ (keep private):
    task08_routing_summary.csv
    task08_country_coverage.csv
    task08_validation_note.txt

Source: Spring 2025 Pew questionnaire, Q30–Q32, Q88–Q89, Q96, Q105.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
OUT = ROOT / 'data' / 'clean'
SAV = RAW / 'Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = RAW / 'Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'

AI_ASK_ALL = [
    'ai_heard', 'ai_cncexc',
    'aireg_us', 'aireg_china', 'aireg_eu', 'aireg_country',
]
NOT_AU = ['age', 'gender', 'internetfreq']
AU_EXAMPLE_COUNTS = {
    'age_australia': 1827,
    'gender_australia': 1827,
    'd_educ_australia': 1821,
}
EXPECTED_COUNTRY_COUNT = 24
EXPECTED_N = 28333
EXPECTED_AU_N = 1829


def nonblank(series: pd.Series) -> pd.Series:
    """Present in original source: distinguish true blanks from coded 8/9/etc."""
    if isinstance(series.dtype, pd.StringDtype) or pd.api.types.is_object_dtype(series.dtype):
        return series.astype('string').str.strip().ne('').fillna(False).astype(bool)
    return series.notna()


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a column-name normalised copy; never edit the input."""
    result = frame.copy(deep=False)
    result.columns = [str(c).strip().lower() for c in frame.columns]
    if not result.columns.is_unique:
        raise AssertionError('Ambiguous or repeated column names after normalisation')
    return result


def australia_code_from_sav(meta: object) -> int:
    """Find Australia by SAV label, without assuming its numeric country code."""
    labels = meta.variable_value_labels['country']
    matches = [int(float(k)) for k, v in labels.items() if str(v).strip().casefold() == 'australia']
    if len(matches) != 1:
        raise AssertionError(f'Expected exactly one Australia country label, got: {matches}')
    return matches[0]


def routing_rule(field: str, scope: str, australia_present: int,
                 other_present: int, sample_n: int, australia_n: int,
                 reference_counts: dict[str, int] | None = None) -> bool:
    """Validate country routing independently of numeric response codes."""
    refs = AU_EXAMPLE_COUNTS if reference_counts is None else reference_counts
    if scope == 'all_countries':
        valid = australia_present + other_present == sample_n
    elif scope == 'outside_australia':
        valid = australia_present == 0 and other_present == sample_n - australia_n
    elif scope == 'australia_only':
        # Conditional questions can be blank in Australia; no responses elsewhere.
        valid = other_present == 0
    else:
        raise ValueError(f'Unknown scope: {scope}')
    if field in refs:
        valid = valid and australia_present == refs[field]
    return bool(valid)


def coverage(frame: pd.DataFrame, source: str, australia_code: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build routing counts; flag violations, not just surprising summary numbers."""
    df = canonical(frame)
    assert df.shape == (EXPECTED_N, 483), f'{source}: unexpected dimensions {df.shape}'
    assert 'country' in df.columns, f'{source}: no country column'
    country = pd.to_numeric(df['country'].astype('string').str.strip(), errors='raise')
    assert country.notna().all(), f'{source}: unassigned countries'
    assert country.nunique() == EXPECTED_COUNTRY_COUNT, f'{source}: expected 24 countries'
    australia = country.eq(australia_code)
    assert int(australia.sum()) == EXPECTED_AU_N, f'{source}: Australia count unexpected'

    au_only = sorted(c for c in df.columns if 'australia' in c)
    required = AI_ASK_ALL + NOT_AU + ['age_australia', 'gender_australia', 'd_educ_australia']
    absent = sorted(set(required) - set(df.columns))
    assert not absent, f'{source}: columns not found: {absent}'
    assert len(au_only) >= 3, f'{source}: cannot locate Australia-specific fields'

    rows = []
    per_country_rows = []
    for field in list(dict.fromkeys(AI_ASK_ALL + NOT_AU + au_only)):
        present = nonblank(df[field])
        au_yes = int((present & australia).sum())
        other_yes = int((present & ~australia).sum())
        expected_scope = ('all_countries' if field in AI_ASK_ALL
                          else 'outside_australia' if field in NOT_AU
                          else 'australia_only')
        # A coded don't know/refused is still a recorded response.
        valid = routing_rule(field, expected_scope, au_yes, other_yes,
                             EXPECTED_N, EXPECTED_AU_N)
        rows.append({
            'source': source, 'variable': field, 'scope': expected_scope,
            'n_australia_present': au_yes,
            'n_other_countries_present': other_yes,
            'n_all_present': int(present.sum()),
            'n_australia_blank': EXPECTED_AU_N - au_yes,
            'n_outside_australia_blank': EXPECTED_N - EXPECTED_AU_N - other_yes,
            'rule_passed': bool(valid),
        })
        counts = (
            pd.DataFrame({'country_code': country.astype(int), 'present': present.astype(int)})
            .groupby('country_code', sort=True, as_index=False)
            .agg(n_answered=('present', 'sum'), n_sample=('present', 'size'))
        )
        for item in counts.itertuples(index=False):
            per_country_rows.append({
                'source': source, 'country_code': int(item.country_code),
                'variable': field, 'n_sample': int(item.n_sample),
                'n_nonblank': int(item.n_answered),
            })
    return pd.DataFrame(rows), pd.DataFrame(per_country_rows)


def cross_check(sav_summary: pd.DataFrame, csv_summary: pd.DataFrame,
                sav_countries: pd.DataFrame, csv_countries: pd.DataFrame) -> None:
    """Assert same routing counts, at both field and individual-country level."""
    s = sav_summary.set_index('variable').sort_index()
    c = csv_summary.set_index('variable').sort_index()
    assert set(s.index) == set(c.index), 'SAV and CSV field lists differ'
    checks = ['n_australia_present', 'n_other_countries_present', 'n_all_present']
    pd.testing.assert_frame_equal(s[checks], c[checks], check_names=False)
    a = sav_countries.set_index(['country_code', 'variable']).sort_index()
    b = csv_countries.set_index(['country_code', 'variable']).sort_index()
    pd.testing.assert_frame_equal(a[['n_sample', 'n_nonblank']], b[['n_sample', 'n_nonblank']])
    assert sav_summary['rule_passed'].all(), (
        'SAV routing violations: ' +
        ', '.join(sav_summary.loc[~sav_summary.rule_passed, 'variable'].tolist())
    )
    assert csv_summary['rule_passed'].all(), 'CSV routing rules were violated'


COUNTRY_ALIASES = {'safrica': 'South Africa', 'skorea': 'South Korea', 'uk': 'UK'}


def country_specific_routing(sav: pd.DataFrame, meta: object) -> pd.DataFrame:
    """Part I d_age_<country> and d_educ_<country> columns: answers only from that country.

    d_age_<country> is the follow-up for people who refused their exact age (98/99),
    so it should also be empty for everyone who gave an age.
    """
    names = meta.variable_value_labels['country']
    code_of = {v: int(k) for k, v in names.items()}
    rows = []
    for col in sorted(c for c in sav.columns if c.startswith(('d_age_', 'd_educ_'))):
        key = col.split('_', 2)[2].split('_')[0]
        country_name = COUNTRY_ALIASES.get(key, key.title())
        code = code_of[country_name]
        in_country = sav['country'].eq(code)
        answered = sav[col].notna()
        row = {'variable': col, 'country': country_name,
               'n_answered_in_country': int((answered & in_country).sum()),
               'n_answered_elsewhere': int((answered & ~in_country).sum()),
               'n_followup_but_age_given': 0}
        if col.startswith('d_age_'):
            row['n_followup_but_age_given'] = int(
                (answered & in_country & ~sav['age'].isin([98, 99])).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def self_test() -> None:
    """Show the blank detector differentiates whitespace from real response codes."""
    sample = pd.Series([' ', '', None, '8', '9', ' 3 ', '0'], dtype='string')
    assert nonblank(sample).tolist() == [False, False, False, True, True, True, True]
    nums = pd.Series([float('nan'), 8, 9, 0], dtype='float64')
    assert nonblank(nums).tolist() == [False, True, True, True]
    # Deliberately violate three routing rules to verify the rule checker.
    assert routing_rule('ai_heard', 'all_countries', 2, 2, 4, 2)
    assert not routing_rule('ai_heard', 'all_countries', 2, 1, 4, 2)
    assert routing_rule('internetfreq', 'outside_australia', 0, 2, 4, 2)
    assert not routing_rule('internetfreq', 'outside_australia', 1, 2, 4, 2)
    assert routing_rule('age_australia', 'australia_only', 1, 0, 4, 2,
                        reference_counts={})
    assert not routing_rule('age_australia', 'australia_only', 1, 1, 4, 2,
                            reference_counts={})
    print('PASS: whitespace-only cells detected as blank')
    print('PASS: nonresponse codes 8 and 9 recognised as recorded values')
    print('PASS: deliberate routing violations correctly detected')
    print('TASK 8 SELF-TEST PASSED')


def main() -> None:
    import pyreadstat
    for path in (SAV, CSV):
        if not path.is_file():
            raise FileNotFoundError(f'Missing original Pew file: {path}')
    print('=== TASK 8: WHO WAS ASKED WHAT? (ORIGINAL FILES) ===')
    sav, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    code = australia_code_from_sav(meta)
    csv = pd.read_csv(CSV, low_memory=False, encoding='utf-8-sig')
    sav_summary, sav_country = coverage(sav, 'SAV', code)
    csv_summary, csv_country = coverage(csv, 'CSV', code)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.concat([sav_summary, csv_summary], ignore_index=True)
    by_country = pd.concat([sav_country, csv_country], ignore_index=True)
    summary.to_csv(OUT / 'task08_routing_summary.csv', index=False)
    by_country.to_csv(OUT / 'task08_country_coverage.csv', index=False)
    (OUT / 'task08_validation_note.txt').write_text(
        'TASK 8 — Questionnaire routing audit (original SAV compared with CSV)\n'
        'Decision: Retain country-specific structural blanks; do not impute answers\n'
        'for questions not asked outside their designated country. Keep Australian\n'
        'age/gender/education blanks distinct from non-applicability elsewhere.\n'
        'Alternative rejected: treating all blanks as individual nonresponse or\n'
        'dropping respondents without internet-frequency answers.\n'
        'Effect: 0 rows changed or removed (this is a validation-only script).\n'
        'AI use: ChatGPT produced the routing audit and independently tested\n'
        'the provided CSV against questionnaire routing rules. SAV execution\n'
        'and review by the student are necessary before signing off.\n'
        'AI errors: none known; record any discovered discrepancy and correction\n'
        'in the separate AI-discrepancy log.\n', encoding='utf-8'
    )

    cols = ['variable', 'scope', 'n_australia_present', 'n_other_countries_present', 'rule_passed']
    print(f'Australia country code from SAV labels: {code}')
    print(f'All-country Q30/Q31/Q32 fields checked: {len(AI_ASK_ALL)}')
    print(f'Australia-specific fields checked: {len(sav_summary) - len(AI_ASK_ALL) - len(NOT_AU)}')
    print('\nSAV routing summary:')
    print(sav_summary[cols].to_string(index=False))
    print('\nCSV routing summary:')
    print(csv_summary[cols].to_string(index=False))
    cross_check(sav_summary, csv_summary, sav_country, csv_country)
    print('\nPASS: Q30/Q31/Q32 present in all 24 countries')
    print('PASS: every *_australia field is blank outside Australia')
    print('PASS: general age, gender, internetfreq populated only outside Australia')
    print('PASS: age_australia 1827/1829; gender_australia 1827/1829; d_educ_australia 1821/1829')
    print('PASS: SAV and CSV agree for each variable in each of 24 countries')

    # Part I country-specific columns (23 d_age_ + 27 d_educ_).
    cs = country_specific_routing(canonical(sav), meta)
    cs.to_csv(OUT / 'task08_country_specific_routing.csv', index=False)
    assert (cs.n_answered_elsewhere == 0).all(), (
        'Answers outside their own country: ' + ', '.join(cs.loc[cs.n_answered_elsewhere > 0, 'variable']))
    print(f'PASS: all {len(cs)} d_age_/d_educ_ columns hold answers only from their own country')
    odd = cs[cs.n_followup_but_age_given > 0]
    for r in odd.itertuples(index=False):
        print(f'NOTE: {r.variable}: {r.n_followup_but_age_given} people answered the age follow-up '
              f'although they gave an exact age (see Task 15)')
    print('Saved: task08_routing_summary.csv, task08_country_coverage.csv, '
          'task08_country_specific_routing.csv, task08_validation_note.txt')
    print('\nTASK 8 PASSED: country routing verified; no data changed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check country-based questionnaire routing')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else main()