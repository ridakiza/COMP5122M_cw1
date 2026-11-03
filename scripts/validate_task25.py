"""COMP5122M P1 Task 25 — remove survey weights, changing NOTHING else.

From repo root:
    python scripts/validate_task25.py --self-test
    python scripts/validate_task25.py

CSV-ONLY PRELIMINARY diagnostic (not equivalent to SAV verification):
    python scripts/validate_task25.py --csv-diagnostic

Illustrative age gap, for each country and each answer:
    % among ages 18-29 MINUS % among ages 65+

Baseline: original Pew weights and every age-group respondent in denominator,
including DK/Refused; scenario: unit weights, SAME denominator and people.
Q30 'A lot' is primary target; Q31's three substantive answers are also shown.
The 24-country mean absolute CHANGE is the simple mean of 24 absolute age-
gap differences (unweighted across countries). These are descriptive numbers,
not significance tests or complex survey design variance estimates.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
try:
    from src.cleaning import do_clean, country_names, awareness_gap
    from validate_task21 import assert_same_by_id, check_independent
except ModuleNotFoundError as exc:
    raise SystemExit(
        'Task 25 requires src/cleaning.py and scripts/validate_task21.py: '
        f'{exc}'
    ) from exc

SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
CLEAN_SNAPSHOT = ROOT / 'data/clean/pew_clean.csv'
OUT = ROOT / 'data/clean'
BANDS = ('18-29', '65+')
EXPECTED_N = 28_333
EXPECTED_Q30_ABS_CHANGE = 2.957393594064276  # ASSERTION ONLY, never input to computation


def numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors='raise')
    else:
        t = s.astype('string').str.strip()
        v = pd.to_numeric(t.mask(t.eq('')), errors='raise')
    assert np.isfinite(v.dropna().to_numpy(dtype=float)).all(), 'Non-finite numeric value'
    return v


def norm_band(s: pd.Series) -> pd.Series:
    return (s.astype('string').str.strip()
            .str.replace('–', '-', regex=False)
            .str.replace('—', '-', regex=False))


def q30_labels(meta: object) -> tuple[int, int, int]:
    """Read A lot/DK/Refused codes directly from original SAV labels."""
    all_labels = getattr(meta, 'variable_value_labels', {}) or {}
    matches = [v for k, v in all_labels.items() if str(k).strip().lower() == 'ai_heard']
    assert len(matches) == 1, 'Original SAV Q30 value labels missing/ambiguous'
    labels = {}
    for k, v in matches[0].items():
        f = float(k)
        assert f.is_integer(), 'Noninteger Q30 response code in SAV'
        labels[int(f)] = str(v).strip().lower().replace('’', "'")
    lot = [k for k, lab in labels.items() if lab == 'a lot']
    dk = [k for k, lab in labels.items() if "don't know" in lab or 'do not know' in lab]
    refused = [k for k, lab in labels.items() if 'refus' in lab]
    assert len(lot) == len(dk) == len(refused) == 1, (
        f'Cannot identify Q30 A lot/DK/Refused from SAV labels: {labels}')
    assert len(labels) == 5 and set(labels) == {1, 2, 3, dk[0], refused[0]}, (
        f'Unexpected Q30 response codes: {labels}')
    assert len({lot[0], dk[0], refused[0]}) == 3
    return lot[0], dk[0], refused[0]


def q31_labels(meta: object) -> tuple[dict[int, str], tuple[int, int]]:
    """Look up substantive answers and both nonresponses BY .SAV LABELS."""
    all_labels = getattr(meta, 'variable_value_labels', {}) or {}
    matches = [v for k, v in all_labels.items() if str(k).strip().lower() == 'ai_cncexc']
    assert len(matches) == 1, 'Original SAV Q31 value labels missing/ambiguous'
    labels = {}
    for k, v in matches[0].items():
        f = float(k)
        assert f.is_integer(), 'Noninteger Q31 response code in SAV'
        labels[int(f)] = str(v).strip().lower().replace('’', "'")
    expected_labels = (
        'more excited than concerned',
        'more concerned than excited',
        'equally concerned and excited',
    )
    answers = {}
    for meaning in expected_labels:
        ids = [k for k, lab in labels.items() if lab == meaning]
        assert len(ids) == 1, f'Missing/duplicate Q31 substantive label {meaning}: {labels}'
        answers[ids[0]] = meaning.title()
    dk = [k for k, lab in labels.items() if "don't know" in lab or 'do not know' in lab]
    refused = [k for k, lab in labels.items() if 'refus' in lab]
    assert len(dk) == len(refused) == 1, f'Cannot identify Q31 nonresponse: {labels}'
    assert len(labels) == 5 and set(answers) | {dk[0], refused[0]} == set(labels), (
        f'Unexpected Q31 codes: {labels}')
    assert len(set(answers) | {dk[0], refused[0]}) == 5
    return answers, (dk[0], refused[0])


def build_codebooks(meta: object) -> dict[str, dict]:
    q30_positive, q30_dk, q30_refused = q30_labels(meta)
    q31_answers, (q31_dk, q31_refused) = q31_labels(meta)
    return {
        'ai_heard': {
            'targets': {q30_positive: 'A lot'},
            'valid_codes': {1, 2, 3, q30_dk, q30_refused},
            'nonresponse': {q30_dk, q30_refused},
        },
        'ai_cncexc': {
            'targets': q31_answers,
            'valid_codes': set(q31_answers) | {q31_dk, q31_refused},
            'nonresponse': {q31_dk, q31_refused},
        },
    }


def calculate(raw: pd.DataFrame, clean: pd.DataFrame, names: dict[int, str],
              specs: dict[str, dict], *, strict: bool = True
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Independent from src.cleaning.awareness_gap; raw codes drive numerators.

    Returns one result per country/answer and one result per country/age/answer.
    """
    required = {'id', 'country', 'weight', *specs}
    assert required.issubset(raw.columns), 'Missing required original data column'
    assert required | {'age_band', 'age_band_recovered'} <= set(clean.columns), (
        'Production cleaning did not create required columns')
    assert len(raw) == len(clean) and len(raw) > 0, 'Rows deleted or added'
    assert raw.id.notna().all() and raw.id.is_unique, 'Raw IDs not unique'
    assert clean.id.notna().all() and clean.id.is_unique, 'Clean IDs not unique'
    assert np.array_equal(numeric(raw.id).to_numpy(dtype=np.int64),
                          numeric(clean.id).to_numpy(dtype=np.int64)), 'ID ordering changed'
    c = numeric(raw.country)
    assert c.notna().all() and c.mod(1).eq(0).all() and c.isin(names).all(), (
        'Country codes invalid / missing from SAV metadata')
    assert np.array_equal(c.to_numpy(float), numeric(clean.country).to_numpy(float)), (
        'Country altered by cleaning')
    w = numeric(raw.weight)
    assert w.notna().all() and w.gt(0).all(), 'Missing/nonpositive survey weights'
    assert np.isclose(w.to_numpy(float), numeric(clean.weight).to_numpy(float),
                      atol=1e-12, rtol=0).all(), 'Weights altered by cleaning'
    band = norm_band(clean.age_band)
    assert band.dropna().isin(('18-29', '30-49', '50-64', '65+')).all(), (
        'Unknown final age-band category')
    assert clean.age_band_recovered.notna().all(), 'Unclassified recovery flags'

    frame = pd.DataFrame({
        'id': numeric(raw.id).to_numpy(dtype=np.int64),
        'country': c.astype(int).map(names).to_numpy(),
        'age_band': band.to_numpy(),
        'weight': w.to_numpy(float),
    })
    for col, spec in specs.items():
        source_q = numeric(raw[col])
        assert source_q.notna().all() and source_q.mod(1).eq(0).all(), (
            f'Unexpected raw missing/noninteger response in {col}')
        assert source_q.isin(spec['valid_codes']).all(), (
            f'{col}: unknown code absent from SAV response labels')
        expected_clean = source_q.mask(source_q.isin(spec['nonresponse']))
        actual_clean = numeric(clean[col])
        assert np.isclose(expected_clean.to_numpy(float), actual_clean.to_numpy(float),
                          rtol=0, atol=0, equal_nan=True).all(), (
            f'{col}: production nonresponses not recoded correctly')
        frame[col] = source_q.to_numpy(dtype=int)

    all_countries = sorted(names.values())
    assert len(all_countries) == len(set(all_countries)), 'Duplicate country names'
    results = []
    group_rows = []
    for country in all_countries:
        country_frame = frame.loc[frame.country.eq(country)]
        for question, spec in specs.items():
            for response_code, response_text in spec['targets'].items():
                pct = {}
                for group in BANDS:
                    sub = country_frame.loc[country_frame.age_band.eq(group)]
                    n = len(sub)
                    assert n > 0, f'Empty age group: {country} {group}'
                    wt_sum = float(sub.weight.sum())
                    assert wt_sum > 0
                    mask = sub[question].eq(response_code)
                    matching_n = int(mask.sum())
                    matching_w = float(sub.loc[mask, 'weight'].sum())
                    # BOTH scenarios include DK and Refused: ONLY weighting changes.
                    weighted_pct = 100.0 * matching_w / wt_sum
                    unweighted_pct = 100.0 * matching_n / n
                    nonresponse_n = int(sub[question].isin(spec['nonresponse']).sum())
                    pct[group] = (weighted_pct, unweighted_pct)
                    group_rows.append({
                        'country': country,
                        'question': question, 'answer_code': response_code,
                        'answer_label': response_text,
                        'age_band': group, 'total_age_group_n': n,
                        'dk_refused_in_denominator_n': nonresponse_n,
                        'target_answer_n': matching_n,
                        'total_age_group_weight': wt_sum,
                        'target_answer_weight': matching_w,
                        'weighted_pct_all_in_denominator': weighted_pct,
                        'unweighted_pct_all_in_denominator': unweighted_pct,
                    })
                weighted_gap = pct['18-29'][0] - pct['65+'][0]
                unweighted_gap = pct['18-29'][1] - pct['65+'][1]
                results.append({
                    'country': country, 'question': question,
                    'answer_code': response_code, 'answer_label': response_text,
                    'baseline_weighted_gap_pp': weighted_gap,
                    'scenario_unweighted_gap_pp': unweighted_gap,
                    'change_pp_unweighted_minus_weighted': unweighted_gap - weighted_gap,
                    'absolute_change_pp': abs(unweighted_gap - weighted_gap),
                })
    r = pd.DataFrame(results)
    g = pd.DataFrame(group_rows)
    assert len(r) == len(all_countries) * 4 and len(g) == len(r) * 2
    if strict:
        assert len(raw) == EXPECTED_N and len(names) == 24, 'Unexpected source dimensions'
        assert int(band.notna().sum()) == 28_148 and int(band.isna().sum()) == 185
        assert int(clean.age_band_recovered.sum()) == 111, 'Expected 111 recoveries'
        assert int(frame.ai_heard.isin(specs['ai_heard']['nonresponse']).sum()) == 808
        assert int(frame.ai_cncexc.isin(specs['ai_cncexc']['nonresponse']).sum()) == 1964
        q30 = r.loc[r.question.eq('ai_heard')]
        assert len(q30) == 24
        abs_mean = float(q30.absolute_change_pp.mean())
        assert np.isclose(abs_mean, EXPECTED_Q30_ABS_CHANGE, atol=1e-6, rtol=0), (
            f'Q30 mean absolute change unexpected: {abs_mean:.9f} pp')
    return r, g


def compare_results(a: pd.DataFrame, b: pd.DataFrame, label: str) -> None:
    """Confirm every country and target answer agree, not just one grand mean."""
    keys = ['country', 'question', 'answer_code']
    x = a.sort_values(keys).reset_index(drop=True)
    y = b.sort_values(keys).reset_index(drop=True)
    assert x[keys].equals(y[keys]), f'{label}: keys differ'
    assert x.answer_label.equals(y.answer_label), f'{label}: answer labels differ'
    for col in (v for v in x.columns if v not in [*keys, 'answer_label']):
        assert np.isclose(x[col].to_numpy(float), y[col].to_numpy(float),
                          atol=1e-9, rtol=0).all(), f'{label}: {col} differs'


def compare_groups(a: pd.DataFrame, b: pd.DataFrame) -> None:
    keys = ['country', 'question', 'answer_code', 'age_band']
    x = a.sort_values(keys).reset_index(drop=True)
    y = b.sort_values(keys).reset_index(drop=True)
    assert x[keys].equals(y[keys]) and x.answer_label.equals(y.answer_label), (
        'Country-age response keys differ between SAV and CSV')
    for col in (v for v in x.columns if v not in [*keys, 'answer_label']):
        assert np.isclose(x[col].to_numpy(float), y[col].to_numpy(float),
                          atol=1e-9, rtol=0).all(), f'SAV/CSV age-group mismatch {col}'


def check_production_awareness(clean: pd.DataFrame, audit: pd.DataFrame) -> None:
    q30 = audit.loc[audit.question.eq('ai_heard')].set_index('country')
    for use_weights, col in ((True, 'baseline_weighted_gap_pp'),
                             (False, 'scenario_unweighted_gap_pp')):
        prod = (awareness_gap(clean, weights=use_weights, keep_dk=True)
                .set_index('country')['gap_pp'].reindex(q30.index))
        assert prod.notna().all()
        assert np.isclose(prod.to_numpy(float), q30[col].to_numpy(float),
                          atol=1e-9, rtol=0).all(), (
            f'Independent raw-Q30 calculations disagree with production for {col}')


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (question, code, answer), part in results.groupby(
            ['question', 'answer_code', 'answer_label'], sort=True):
        largest = part.loc[part.absolute_change_pp.idxmax()]
        rows.append({
            'question': question, 'answer_code': int(code), 'answer_label': answer,
            'countries': len(part),
            'mean_absolute_change_pp': float(part.absolute_change_pp.mean()),
            'mean_signed_change_pp': float(part.change_pp_unweighted_minus_weighted.mean()),
            'max_abs_change_country': largest.country,
            'max_signed_change_pp': float(largest.change_pp_unweighted_minus_weighted),
            'age_gap_definition': '18-29 minus 65+',
            'baseline': 'Pew-weighted, ALL respondents including DK+Refused',
            'scenario': 'unweighted, SAME people and denominator',
        })
    return pd.DataFrame(rows)


def must_fail(fn, label: str) -> None:
    try:
        fn()
    except AssertionError:
        print('PASS:', label)
    else:
        raise AssertionError(f'Self-test did not catch: {label}')


def self_test() -> None:
    meta = SimpleNamespace(variable_value_labels={
        'ai_heard': {1: 'A lot', 2: 'A little', 3: 'Nothing at all',
                     8: "Don't know", 9: 'Refused'},
        'ai_cncexc': {1: 'More excited than concerned',
                      2: 'More concerned than excited',
                      3: 'Equally concerned and excited',
                      8: "Don't know", 9: 'Refused'},
    })
    specs = build_codebooks(meta)
    raw = pd.DataFrame({
        'id': range(10, 18), 'country': [5] * 8,
        'weight': [3., 1., 1., 1., 1., 1., 2., 2.],
        'ai_heard': [1, 8, 1, 9, 1, 2, 2, 8],
        'ai_cncexc': [1, 8, 2, 3, 2, 2, 9, 1],
    })
    clean = raw.copy(deep=True)
    for col, spec in specs.items():
        clean[col] = numeric(raw[col]).mask(numeric(raw[col]).isin(spec['nonresponse']))
    clean['age_band'] = ['18–29'] * 4 + ['65+'] * 4
    clean['age_band_recovered'] = False
    raw_before = raw.copy(deep=True)
    c_before = clean.copy(deep=True)
    r, g = calculate(raw, clean, {5: 'Toy'}, specs, strict=False)
    q = r.loc[r.question.eq('ai_heard')].iloc[0]
    assert np.isclose(q.baseline_weighted_gap_pp, 50.0)
    assert np.isclose(q.scenario_unweighted_gap_pp, 25.0)
    assert np.isclose(q.change_pp_unweighted_minus_weighted, -25.0)
    assert len(r) == 4 and len(g) == 8
    young = g.loc[g.question.eq('ai_heard') & g.age_band.eq('18-29')].iloc[0]
    assert (young.total_age_group_n, young.dk_refused_in_denominator_n,
            young.target_answer_n) == (4, 2, 2)
    assert np.isclose(young.weighted_pct_all_in_denominator, 100 * 4 / 6)
    assert np.isclose(young.unweighted_pct_all_in_denominator, 50.0)
    pd.testing.assert_frame_equal(raw, raw_before, check_exact=True)
    pd.testing.assert_frame_equal(clean, c_before, check_exact=True)
    print('PASS: synthetic Q30 gap uses weighted vs unit weights, SAME DK-inclusive denominator')
    print('PASS: Q31 substantive answers compared; no changes to original data')
    wrong = raw.copy(deep=True); wrong.loc[0, 'ai_cncexc'] = 7
    must_fail(lambda: calculate(wrong, clean, {5: 'Toy'}, specs, strict=False),
              'unlabelled Q31 code rejected')
    wrong = clean.copy(deep=True); wrong.loc[1, 'ai_cncexc'] = 8
    must_fail(lambda: calculate(raw, wrong, {5: 'Toy'}, specs, strict=False),
              'incorrect Q31 DK recoding detected')
    wrong = raw.copy(deep=True); wrong.loc[0, 'weight'] = 0
    must_fail(lambda: calculate(wrong, clean, {5: 'Toy'}, specs, strict=False),
              'nonpositive survey weight detected')
    wrong = clean.copy(deep=True); wrong.loc[0, 'age_band'] = 'unknown'
    must_fail(lambda: calculate(raw, wrong, {5: 'Toy'}, specs, strict=False),
              'unknown age band rejected')
    wrong = clean.copy(deep=True); wrong.loc[0, 'id'] = 999
    must_fail(lambda: calculate(raw, wrong, {5: 'Toy'}, specs, strict=False),
              'misaligned respondent IDs detected')
    wrong_meta = SimpleNamespace(variable_value_labels={**meta.variable_value_labels,
        'ai_cncexc': {1: 'More excited than concerned',
                      2: 'More concerned than excited', 3: 'Equally concerned and excited',
                      8: 'Unclear', 9: 'Refused'}})
    must_fail(lambda: build_codebooks(wrong_meta), 'unknown Q31 nonresponse label rejected')
    print('TASK 25 SELF-TEST PASSED')


def diagnostic() -> None:
    """CSV + existing clean snapshot ONLY; not SAV-based source-label check."""
    assert CSV.is_file() and CLEAN_SNAPSHOT.is_file(), 'Original CSV/clean snapshot missing'
    raw = pd.read_csv(CSV, low_memory=False)
    clean = pd.read_csv(CLEAN_SNAPSHOT, low_memory=False)
    assert raw.shape == (EXPECTED_N, 483) and len(clean) == EXPECTED_N
    names = {int(k): str(v) for k, v in clean[['country', 'country_name']]
             .drop_duplicates().itertuples(index=False)}
    assert len(names) == 24
    fake_meta = SimpleNamespace(variable_value_labels={
        'ai_heard': {1: 'A lot', 2: 'A little', 3: 'Nothing at all',
                     8: "Don't know", 9: 'Refused'},
        'ai_cncexc': {1: 'More excited than concerned',
                      2: 'More concerned than excited',
                      3: 'Equally concerned and excited',
                      8: "Don't know", 9: 'Refused'},
    })
    specs = build_codebooks(fake_meta)
    assert_same_by_id(raw, clean, ['country', 'weight'],
                      context='CSV original vs cleaned snapshot')
    res, grp = calculate(raw, clean, names, specs)
    check_production_awareness(clean, res)
    s = summarise(res)
    print('=== TASK 25 CSV-ONLY DIAGNOSTIC (NOT SAV VERIFICATION) ===')
    print(s[['question', 'answer_label', 'mean_absolute_change_pp',
             'mean_signed_change_pp', 'max_abs_change_country',
             'max_signed_change_pp']].to_string(index=False,
                 float_format=lambda x: f'{x:.6f}'))
    print('PASS: 24 country Q30 changes and 72 Q31 changes independently calculated')
    print('NOTE: original SAV label metadata and SAV/CSV cross-check still needed.')


def main() -> None:
    try:
        import pyreadstat
    except ImportError as exc:
        raise SystemExit('Install pyreadstat in the active venv to validate the SAV') from exc
    assert SAV.is_file() and CSV.is_file(), 'Missing source files in data/raw/'
    print('=== TASK 25: REMOVE WEIGHTS — Q30/Q31 AGE-GAP SENSITIVITY ===')
    raw_sav, meta = pyreadstat.read_sav(
        str(SAV), apply_value_formats=False, user_missing=True)
    raw_csv = pd.read_csv(CSV, encoding='utf-8-sig', low_memory=False)
    raw_sav.columns = raw_sav.columns.str.strip().str.lower()
    raw_csv.columns = raw_csv.columns.str.strip().str.lower()
    assert raw_sav.shape == raw_csv.shape == (EXPECTED_N, 483)
    specs = build_codebooks(meta)
    names = country_names(meta)
    print(f'SAV Q30 value-label codes: response={list(specs["ai_heard"]["targets"])}; '
          f'nonresponse={sorted(specs["ai_heard"]["nonresponse"])}')
    print(f'SAV Q31 value-label codes: responses={sorted(specs["ai_cncexc"]["targets"])}; '
          f'nonresponse={sorted(specs["ai_cncexc"]["nonresponse"])}')
    assert_same_by_id(raw_sav, raw_csv,
                      ['country','weight','age','age_australia','ai_heard','ai_cncexc'],
                      context='Original SAV vs CSV')
    raw_sav_before = raw_sav.copy(deep=True)
    raw_csv_before = raw_csv.copy(deep=True)
    prod_sav = do_clean(raw_sav, meta, log=[])
    prod_csv = do_clean(raw_csv, meta, log=[])
    pd.testing.assert_frame_equal(raw_sav, raw_sav_before, check_exact=True)
    pd.testing.assert_frame_equal(raw_csv, raw_csv_before, check_exact=True)
    print('PASS: production cleaning did not mutate original SAV/CSV DataFrames')
    assert_same_by_id(prod_sav, prod_csv,
                      ['country', 'weight', 'ai_heard', 'ai_cncexc',
                       'age_band', 'age_band_recovered'],
                      context='Cleaned SAV vs CSV')
    check_independent(raw_sav, prod_sav, meta, name='SAV')
    check_independent(raw_csv, prod_csv, meta, name='CSV')
    sav_res, sav_groups = calculate(raw_sav, prod_sav, names, specs)
    csv_res, csv_groups = calculate(raw_csv, prod_csv, names, specs)
    compare_results(sav_res, csv_res, 'SAV vs CSV across all country-response gaps')
    compare_groups(sav_groups, csv_groups)
    check_production_awareness(prod_sav, sav_res)
    check_production_awareness(prod_csv, csv_res)
    print('PASS: 24/24 Q30 and 72/72 Q31 country-response sensitivity comparisons match SAV/CSV')
    print('PASS: 192/192 country-age-question-response numerators and denominators match SAV/CSV')
    print('PASS: independent raw Q30 calculations agree with production awareness_gap()')

    s = summarise(sav_res)
    print('\nMean absolute age-gap change on REMOVING WEIGHTS (pp):')
    print(s[['question', 'answer_label', 'mean_absolute_change_pp',
             'mean_signed_change_pp', 'max_abs_change_country',
             'max_signed_change_pp']].to_string(index=False,
                 float_format=lambda x: f'{x:.6f}'))
    q30 = sav_res.loc[sav_res.question.eq('ai_heard')].sort_values('country')
    print('\nQ30: A lot — country changes (unweighted gap MINUS weighted gap, pp):')
    print(q30[['country', 'baseline_weighted_gap_pp',
               'scenario_unweighted_gap_pp', 'change_pp_unweighted_minus_weighted']]
          .to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    mean_abs = float(q30.absolute_change_pp.mean())
    largest = q30.loc[q30.absolute_change_pp.idxmax()]
    assert len(raw_sav) == len(prod_sav) == len(prod_csv) == EXPECTED_N
    assert int(prod_sav.age_band.isna().sum()) == 185
    print('\nPASS: 28,333 respondents preserved, 185 age bands unknown; both scenarios')
    print('      retain original DK/refused in the denominators; ONLY weights change.')

    OUT.mkdir(parents=True, exist_ok=True)
    sav_res.loc[sav_res.question.eq('ai_heard')].to_csv(
        OUT / 'task25_unweighted_q30_country_audit.csv', index=False)
    sav_res.loc[sav_res.question.eq('ai_cncexc')].to_csv(
        OUT / 'task25_unweighted_q31_country_audit.csv', index=False)
    sav_groups.to_csv(OUT / 'task25_weighting_age_group_audit.csv', index=False)
    s.to_csv(OUT / 'task25_sensitivity_summary.csv', index=False)
    pd.DataFrame([{
        'task': 25,
        'decision': 'Use supplied Pew survey weights in main estimates; compare with unit-weighted calculations',
        'alternative': 'Ignore survey weights throughout',
        'measure': 'Country-specific weighted vs unweighted gap (age 18-29 minus 65+) for Q30 A lot; also three Q31 answers',
        'other_choices_fixed': 'Same respondents and same DK/refused-inclusive denominator in both scenarios',
        'effect': f'Q30 mean absolute country-gap change {mean_abs:.6f} pp; largest {largest.country} {largest.change_pp_unweighted_minus_weighted:+.6f} pp',
        'caution': 'Descriptive sensitivity, not significance test; unweighted does not represent Pew population estimates',
        'verification': 'Original SAV labels; SAV/CSV by ID; independent ages and source-code numerators; production Q30 comparison',
        'status': 'VERIFIED_WHEN_SCRIPT_PASSES',
    }]).to_csv(OUT / 'task25_decision_log_entry.csv', index=False)
    pd.DataFrame([{
        'task': 25,
        'ai_contribution': 'Created source-label-driven Q30/Q31 weighting sensitivity and validation script, tests and audit exports',
        'verification': 'CSV diagnostic and synthetic tests by AI; original SAV and CSV and independent age comparison on student run',
        'source': 'Original Pew survey SAV and CSV; previously verified P1 cleaning and age recovery',
        'ai_errors_detected': 'None identified in delivered code during tests',
    }]).to_csv(OUT / 'task25_ai_use_log_entry.csv', index=False)
    (OUT / 'task25_validation_note.txt').write_text(
        'TASK 25: WEIGHTING SENSITIVITY\n'
        'Measure: per-country, (18-29 %) minus (65+ %), for Q30 A lot and Q31 '
        'More Excited/More Concerned/Equally. Baseline percentages use Pew survey '
        'weights. Scenario percentages assign weight 1 to each respondent. '
        'Both use identical age groups and DK/Refused-INCLUSIVE denominators. '
        'No rows or source values are removed or changed. Original SAV value '
        'labels are read at runtime. Every country-answer and country-age-answer '
        'output is recomputed from original raw responses and compared SAV vs CSV; '
        'Q30 results also compared with src.cleaning.awareness_gap().\n'
        f'Q30 mean absolute country age-gap difference: {mean_abs:.6f} pp. '
        f'Largest individual country: {largest.country}, '
        f'{largest.change_pp_unweighted_minus_weighted:+.6f} pp.\n'
        'The mean absolute difference across 24 countries is a descriptive '
        'unweighted mean of country changes. Neither a statistical significance '
        'test nor an estimator-specific full survey design effect.\n', encoding='utf-8')
    print('\nSaved: task25_unweighted_q30_country_audit.csv,')
    print('       task25_unweighted_q31_country_audit.csv, task25_weighting_age_group_audit.csv,')
    print('       task25_sensitivity_summary.csv, task25_decision_log_entry.csv,')
    print('       task25_ai_use_log_entry.csv, task25_validation_note.txt')
    print('TASK 25 PASSED: removing weights changes Q30 country age gaps by')
    print(f'                {mean_abs:.6f} pp absolute mean; 24 Q30 + 72 Q31 results agree SAV/CSV;')
    print('                identical DK-inclusive denominators; no rows removed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    opts = parser.add_mutually_exclusive_group()
    opts.add_argument('--self-test', action='store_true')
    opts.add_argument('--csv-diagnostic', action='store_true')
    arg = parser.parse_args()
    if arg.self_test:
        self_test()
    elif arg.csv_diagnostic:
        diagnostic()
    else:
        main()
