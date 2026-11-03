"""COMP5122M P1 Task 26: sensitivity to recovered follow-up age bands.

Run in the repository root:
    python scripts/validate_task26.py --self-test
    python scripts/validate_task26.py

Optional independent CSV-only preliminary check (NOT original SAV verification):
    python scripts/validate_task26.py --csv-diagnostic

Baseline: Pew-weighted Q30 'A lot' (18-29 minus 65+) by country, with all
recovered age bands. Scenario: set the age_band of the 111 respondents whose
age was unambiguously recovered from country follow-ups to MISSING; do NOT
remove records. Keep original weights and DK/Refused-inclusive denominators.
An additional Q31 three-response sensitivity is provided for consistency with
Tasks 24/25. These are descriptive comparisons, not statistical tests.
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
    from validate_task25 import build_codebooks, numeric, norm_band
except ModuleNotFoundError as exc:
    raise SystemExit(f'Task 26 requires src/cleaning.py, validate_task21.py and validate_task25.py: {exc}') from exc

SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
SNAPSHOT = ROOT / 'data/clean/pew_clean.csv'
OUT = ROOT / 'data/clean'
N_EXPECTED = 28_333
VALID_BANDS = ('18-29', '30-49', '50-64', '65+')
GAP_BANDS = ('18-29', '65+')
TARGET_Q30_MEAN_ABS = 0.205652535726  # POST-COMPUTATION ASSERTION ONLY


def make_without_recovered(clean: pd.DataFrame) -> pd.DataFrame:
    """Only remove recoveries' *age classification*; NEVER delete a person."""
    assert {'id', 'age_band', 'age_band_recovered'}.issubset(clean.columns)
    assert clean.id.notna().all() and clean.id.is_unique
    flags = clean.age_band_recovered
    assert flags.notna().all() and flags.isin([True, False]).all(), 'Bad recovery indicator'
    recovered = flags.astype(bool)
    assert not norm_band(clean.loc[recovered, 'age_band']).isna().any(), (
        'Recovered age classification is missing from baseline')
    scenario = clean.copy(deep=True)
    scenario.loc[recovered, 'age_band'] = np.nan
    assert len(scenario) == len(clean)
    assert scenario.id.equals(clean.id)
    assert int(scenario.age_band.isna().sum()) == int(clean.age_band.isna().sum()) + int(recovered.sum())
    assert norm_band(scenario.loc[~recovered, 'age_band']).fillna('<NA>').equals(
        norm_band(clean.loc[~recovered, 'age_band']).fillna('<NA>'))
    assert (scenario.loc[recovered, 'age_band'].isna()).all()
    # Demonstrate no changes to ANY other fields.
    for col in clean.columns:
        if col != 'age_band':
            pd.testing.assert_series_equal(clean[col], scenario[col], check_exact=True)
    return scenario


def calculate(raw: pd.DataFrame, clean: pd.DataFrame,
              names: dict[int, str], specs: dict[str, dict], *,
              strict: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild percentages from untouched source responses; no production gap reuse."""
    needed = {'id', 'country', 'weight', *specs}
    assert needed.issubset(raw.columns)
    assert needed | {'age_band', 'age_band_recovered', 'country_name'} <= set(clean.columns)
    assert len(raw) == len(clean), 'Source rows deleted or added'
    assert raw.id.notna().all() and raw.id.is_unique and clean.id.is_unique
    assert np.array_equal(numeric(raw.id).to_numpy(dtype=np.int64),
                          numeric(clean.id).to_numpy(dtype=np.int64)), 'IDs changed/order changed'
    c = numeric(raw.country)
    assert c.notna().all() and c.isin(names).all(), 'Country not present in SAV metadata'
    assert np.array_equal(c.to_numpy(float), numeric(clean.country).to_numpy(float))
    source_countries = c.astype(int).map(names)
    assert source_countries.astype('string').equals(clean.country_name.astype('string')), (
        'Clean country names disagree with original SAV metadata')
    w = numeric(raw.weight)
    assert w.notna().all() and w.gt(0).all(), 'Missing or nonpositive weight'
    assert np.isclose(w.to_numpy(float), numeric(clean.weight).to_numpy(float),
                      rtol=0, atol=1e-12).all(), 'Production weights changed'
    flags = clean.age_band_recovered
    assert flags.notna().all() and flags.isin([True, False]).all()
    flags = flags.astype(bool)
    bands = norm_band(clean.age_band)
    assert bands.dropna().isin(VALID_BANDS).all(), 'Unknown age band'
    assert not bands[flags].isna().any(), 'Recovered band absent in baseline'
    if strict:
        assert len(raw) == N_EXPECTED, 'Unexpected survey sample size'
        assert int(flags.sum()) == 111, 'Incorrect recovered age-band count'
        assert int(bands.isna().sum()) == 185, 'Incorrect missing age-band count'
        assert len(names) == 24, 'Expected 24 country names'
        assert int(bands[flags].eq('18-29').sum()) == 15
        assert int(bands[flags].eq('65+').sum()) == 31

    scenario = make_without_recovered(clean)
    expected_baseline_known = int(bands.notna().sum())
    expected_scenario_known = int(scenario.age_band.notna().sum())
    assert expected_baseline_known - expected_scenario_known == int(flags.sum())
    frame = pd.DataFrame({
        'country': source_countries.to_numpy(),
        'age_band': bands.to_numpy(),
        'recovered': flags.to_numpy(bool),
        'weight': w.to_numpy(float),
    })
    for q, spec in specs.items():
        original = numeric(raw[q])
        assert original.notna().all() and original.mod(1).eq(0).all(), (
            f'Invalid original {q} responses')
        assert original.isin(spec['valid_codes']).all(), f'Unknown original response code in {q}'
        expected_clean = original.mask(original.isin(spec['nonresponse']))
        actual_clean = numeric(clean[q])
        assert np.isclose(expected_clean.to_numpy(float), actual_clean.to_numpy(float),
                          rtol=0, atol=0, equal_nan=True).all(), (
            f'Production {q} cleaning differs from original SAV-label rule')
        frame[q] = original.to_numpy(dtype=int)

    group_rows = []
    country_rows = []
    for country in sorted(names.values()):
        country_data = frame.loc[frame.country.eq(country)]
        assert len(country_data) > 0
        for question, spec in specs.items():
            for target_code, target_label in spec['targets'].items():
                gaps = {}
                for mode in ('baseline', 'without_recovered'):
                    percentages = {}
                    for age in GAP_BANDS:
                        mask = country_data.age_band.eq(age)
                        if mode == 'without_recovered':
                            mask &= ~country_data.recovered
                        group = country_data.loc[mask]
                        count = len(group)
                        assert count > 0, f'Empty group {country} {age} {mode}'
                        weighted_den = float(group.weight.sum())
                        assert weighted_den > 0
                        yes = group[question].eq(target_code)
                        weighted_num = float(group.loc[yes, 'weight'].sum())
                        yes_n = int(yes.sum())
                        nonresponse_n = int(group[question].isin(spec['nonresponse']).sum())
                        percentage = weighted_num / weighted_den * 100
                        percentages[age] = percentage
                        group_rows.append({
                            'country': country, 'question': question,
                            'answer_code': int(target_code), 'answer_label': target_label,
                            'scenario': mode, 'age_band': age,
                            'age_group_n': count,
                            'recovered_in_group_n': int(group.recovered.sum()),
                            'dk_refused_in_denominator_n': nonresponse_n,
                            'target_answer_n': yes_n,
                            'sum_weight_denominator': weighted_den,
                            'sum_target_answer_weight': weighted_num,
                            'weighted_percent': percentage,
                        })
                    gaps[mode] = percentages[GAP_BANDS[0]] - percentages[GAP_BANDS[1]]
                change = gaps['without_recovered'] - gaps['baseline']
                country_rows.append({
                    'country': country, 'question': question,
                    'answer_code': int(target_code), 'answer_label': target_label,
                    'baseline_gap_pp': gaps['baseline'],
                    'without_recovered_gap_pp': gaps['without_recovered'],
                    'change_pp': change, 'absolute_change_pp': abs(change),
                })
    results = pd.DataFrame(country_rows).sort_values(
        ['question', 'answer_code', 'country']).reset_index(drop=True)
    groups = pd.DataFrame(group_rows).sort_values(
        ['question','answer_code','country','scenario','age_band']).reset_index(drop=True)
    # Country-level recovery provenance audit, independent of Q30/Q31 responses.
    provenance = []
    for country in sorted(names.values()):
        g = frame.loc[frame.country.eq(country)]
        provenance.append({
            'country': country, 'total_respondents': len(g),
            'recovered_18_29': int((g.recovered & g.age_band.eq('18-29')).sum()),
            'recovered_30_49': int((g.recovered & g.age_band.eq('30-49')).sum()),
            'recovered_50_64': int((g.recovered & g.age_band.eq('50-64')).sum()),
            'recovered_65_plus': int((g.recovered & g.age_band.eq('65+')).sum()),
            'recovered_total': int(g.recovered.sum()),
            'baseline_missing_age': int(g.age_band.isna().sum()),
            'scenario_missing_age': int(g.age_band.isna().sum() + g.recovered.sum()),
        })
    recoveries = pd.DataFrame(provenance)
    assert int(recoveries.recovered_total.sum()) == int(flags.sum())
    if strict:
        assert len(results) == 96 and len(groups) == 384
        assert int(recoveries.scenario_missing_age.sum()) == 296
        primary = results.loc[results.question.eq('ai_heard')]
        assert np.isclose(primary.absolute_change_pp.mean(), TARGET_Q30_MEAN_ABS,
                          rtol=0, atol=1e-8), 'Q30 country-mean sensitivity is unexpected'
    return results, groups, recoveries


def compare_numeric_tables(a: pd.DataFrame, b: pd.DataFrame,
                           keys: list[str], context: str) -> None:
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    assert len(a) == len(b)
    for col in keys:
        assert a[col].astype('string').equals(b[col].astype('string')), (
            f'{context}: key mismatch in {col}')
    for col in a.columns:
        if col in keys:
            continue
        assert col in b.columns
        av = numeric(a[col]).to_numpy(dtype=float)
        bv = numeric(b[col]).to_numpy(dtype=float)
        assert np.isclose(av, bv, rtol=0, atol=1e-9,
                          equal_nan=True).all(), f'{context}: mismatch in {col}'
    print(f'PASS: {context} — {len(a)} rows agree')


def check_production(clean: pd.DataFrame, results: pd.DataFrame) -> None:
    sub = results.loc[results.question.eq('ai_heard')].set_index('country')
    modes = ((False, 'baseline_gap_pp'), (True, 'without_recovered_gap_pp'))
    for excl, col in modes:
        test = awareness_gap(clean, weights=True, keep_dk=True,
                             exclude_recovered=excl).set_index('country')['gap_pp']
        test = test.reindex(sub.index)
        assert test.notna().all() and np.isclose(
            test.to_numpy(float), sub[col].to_numpy(float),
            rtol=0, atol=1e-9).all(), (
            f'Independent Q30 calculations differ from production awareness_gap ({col})')
    print('PASS: independent raw Q30 gaps match production awareness_gap() in both scenarios')


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    data=[]
    for (q, answer_code, label), g in results.groupby(
            ['question','answer_code','answer_label'], sort=True):
        biggest = g.loc[g.absolute_change_pp.idxmax()]
        data.append({
            'question': q, 'answer_code': int(answer_code), 'answer_label': label,
            'countries': len(g),
            'mean_absolute_change_pp': float(g.absolute_change_pp.mean()),
            'mean_signed_change_pp': float(g.change_pp.mean()),
            'countries_with_nonzero_change': int(g.absolute_change_pp.gt(1e-9).sum()),
            'max_change_country': biggest.country,
            'max_signed_change_pp': float(biggest.change_pp),
        })
    return pd.DataFrame(data)


def should_fail(action, description: str) -> None:
    try:
        action()
    except AssertionError:
        print('PASS:', description)
    else:
        raise AssertionError(f'Failed to catch intentional error: {description}')


def self_test() -> None:
    meta = SimpleNamespace(variable_value_labels={
        'ai_heard': {1:'A lot', 2:'A little', 3:'Nothing at all', 8:"Don't know", 9:'Refused'},
        'ai_cncexc': {1:'More excited than concerned',2:'More concerned than excited',
                      3:'Equally concerned and excited',8:"Don't know",9:'Refused'},
    })
    specs = build_codebooks(meta)
    raw = pd.DataFrame({
        'id': range(1,9), 'country': [5]*8,
        'weight': [2.,1.,1.,1.,1.,1.,1.,1.],
        'ai_heard': [1,1,8,2,1,2,1,9],
        'ai_cncexc': [1,2,8,3,1,2,3,9],
    })
    clean = raw.copy(deep=True)
    for q,s in specs.items():
        clean[q] = numeric(raw[q]).mask(numeric(raw[q]).isin(s['nonresponse']))
    clean['country_name'] = 'Toy'
    clean['age_band'] = ['18–29']*4+['65+']*4
    clean['age_band_recovered'] = [False,True,False,False,False,False,True,False]
    before_raw = raw.copy(deep=True)
    before_clean = clean.copy(deep=True)
    result, groups, provenance = calculate(raw, clean, {5:'Toy'}, specs, strict=False)
    assert len(result)==4 and len(groups)==16 and int(provenance.recovered_total.sum())==2
    q30 = result.loc[result.question.eq('ai_heard')].iloc[0]
    # Young baseline: (2+1)/(2+1+1+1)=3/5; oldest: 2/4.
    # Young without recovered: 2/4; oldest without recovered: 1/3.
    assert np.isclose(q30.baseline_gap_pp,10)
    assert np.isclose(q30.without_recovered_gap_pp,100*(.5-1/3))
    assert np.isclose(q30.change_pp,100*(.5-1/3)-10)
    alternative = make_without_recovered(clean)
    assert len(alternative)==8 and len(raw)==8 and int(alternative.age_band.isna().sum())==2
    pd.testing.assert_frame_equal(raw,before_raw,check_exact=True)
    pd.testing.assert_frame_equal(clean,before_clean,check_exact=True)
    print('PASS: recovered age band becomes unknown, but original records and other fields unchanged')
    print('PASS: weighted 18–29 minus 65+ gaps match hand-calculated synthetic example')
    corrupted = clean.copy()
    corrupted.loc[corrupted.id.eq(2),'age_band_recovered'] = False
    corrupted.loc[corrupted.id.eq(7),'age_band_recovered'] = False
    changed, _, _ = calculate(raw,corrupted,{5:'Toy'},specs,strict=False)
    assert not np.isclose(changed.iloc[0].change_pp,q30.change_pp)
    print('PASS: removing recovery provenance changes the sensitivity estimate')
    wrong = raw.copy();wrong.loc[0,'weight']=-1
    should_fail(lambda: calculate(wrong,clean,{5:'Toy'},specs,strict=False),
                'nonpositive survey weight detected')
    wrong = raw.copy();wrong.loc[0,'ai_heard']=77
    should_fail(lambda: calculate(wrong,clean,{5:'Toy'},specs,strict=False),
                'unlabelled original response code detected')
    wrong = clean.copy();wrong.loc[0,'ai_heard']=8
    should_fail(lambda: calculate(raw,wrong,{5:'Toy'},specs,strict=False),
                'failure to clean Q30 nonresponse detected')
    wrong=clean.iloc[:-1].copy()
    should_fail(lambda: calculate(raw,wrong,{5:'Toy'},specs,strict=False),
                'dropped respondent detected')
    wrong=clean.copy();wrong.loc[wrong.id.eq(2),'age_band']=np.nan
    should_fail(lambda: make_without_recovered(wrong),
                'missing recovered baseline band detected')
    bad_meta=SimpleNamespace(variable_value_labels={
        **meta.variable_value_labels,
        'ai_cncexc': {1:'More excited than concerned',2:'More concerned than excited',
                       3:'Equally concerned and excited',8:'Other',9:'Refused'},
    })
    should_fail(lambda: build_codebooks(bad_meta),
                'ambiguous nonresponse label rejected')
    print('TASK 26 SELF-TEST PASSED')


def diagnostic() -> None:
    assert CSV.is_file() and SNAPSHOT.is_file(), 'CSV or cleaned snapshot missing'
    raw = pd.read_csv(CSV, low_memory=False)
    cleaned = pd.read_csv(SNAPSHOT, low_memory=False)
    assert len(raw)==len(cleaned)==N_EXPECTED
    names = dict(cleaned[['country','country_name']].drop_duplicates().itertuples(index=False,name=None))
    names = {int(k):str(v) for k,v in names.items()}
    dummy = SimpleNamespace(variable_value_labels={
        'ai_heard': {1:'A lot',2:'A little',3:'Nothing at all',8:"Don't know",9:'Refused'},
        'ai_cncexc': {1:'More excited than concerned',2:'More concerned than excited',
                      3:'Equally concerned and excited',8:"Don't know",9:'Refused'},
    })
    specs = build_codebooks(dummy)
    assert_same_by_id(raw, cleaned, ['country','weight'], context='CSV raw vs cleaned snapshot')
    res, groups, provenance = calculate(raw,cleaned,names,specs)
    check_production(cleaned,res)
    summary = summarise(res)
    print('=== TASK 26: CSV-ONLY DIAGNOSTIC (NOT SAV VERIFICATION) ===')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print(f'Baseline age bands known = {int(cleaned.age_band.notna().sum()):,}; '
          f'scenario known = {int(cleaned.age_band.notna().sum()-cleaned.age_band_recovered.sum()):,}')
    print('NOTE: source value labels above are synthetic proxies; use original SAV for authoritative check')


def main() -> None:
    try:
        import pyreadstat
    except ImportError as exc:
        raise SystemExit('Install pyreadstat in active venv to validate the original SAV') from exc
    assert SAV.is_file() and CSV.is_file(), 'Missing original SAV/CSV in data/raw/'
    print('=== TASK 26: EXCLUDE RECOVERED AGE BANDS — SENSITIVITY ===')
    sav_raw, meta = pyreadstat.read_sav(
        str(SAV), apply_value_formats=False, user_missing=True)
    csv_raw = pd.read_csv(CSV, encoding='utf-8-sig',low_memory=False)
    sav_raw.columns = sav_raw.columns.str.strip().str.lower()
    csv_raw.columns = csv_raw.columns.str.strip().str.lower()
    assert sav_raw.shape==csv_raw.shape==(N_EXPECTED,483), 'Unexpected source shape'
    names=country_names(meta)
    specs=build_codebooks(meta)
    print('SAV Q30 target and nonresponses:',specs['ai_heard']['targets'],sorted(specs['ai_heard']['nonresponse']))
    print('SAV Q31 targets and nonresponses:',specs['ai_cncexc']['targets'],sorted(specs['ai_cncexc']['nonresponse']))
    assert_same_by_id(sav_raw,csv_raw,
                      ['country','weight','age','age_australia','ai_heard','ai_cncexc'],
                      context='Original SAV vs CSV')
    sav_before=sav_raw.copy(deep=True)
    csv_before=csv_raw.copy(deep=True)
    sav_clean=do_clean(sav_raw,meta,log=[])
    csv_clean=do_clean(csv_raw,meta,log=[])
    pd.testing.assert_frame_equal(sav_raw,sav_before,check_exact=True)
    pd.testing.assert_frame_equal(csv_raw,csv_before,check_exact=True)
    print('PASS: original SAV/CSV source frames unchanged')
    assert_same_by_id(sav_clean,csv_clean,
                      ['country','weight','age_band','age_band_recovered','ai_heard','ai_cncexc'],
                      context='Cleaned SAV vs CSV')
    check_independent(sav_raw,sav_clean,meta,name='SAV')
    check_independent(csv_raw,csv_clean,meta,name='CSV')
    sav_res,sav_groups,sav_provenance=calculate(sav_raw,sav_clean,names,specs)
    csv_res,csv_groups,csv_provenance=calculate(csv_raw,csv_clean,names,specs)
    compare_numeric_tables(sav_res,csv_res,
                           ['country','question','answer_code','answer_label'],
                           'all 96 country-response gaps across SAV/CSV')
    compare_numeric_tables(sav_groups,csv_groups,
                           ['country','question','answer_code','answer_label','scenario','age_band'],
                           'all 384 group-level numerator and denominator checks across SAV/CSV')
    compare_numeric_tables(sav_provenance,csv_provenance,
                           ['country'],'all 24 country recovery counts across SAV/CSV')
    check_production(sav_clean,sav_res)
    check_production(csv_clean,csv_res)
    s=summarise(sav_res)
    print('\nMean absolute age-gap change from EXCLUDING RECOVERED BANDS (pp):')
    print(s.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    q30=sav_res.loc[sav_res.question.eq('ai_heard')].sort_values('country')
    print('\nQ30 country changes: (WITHOUT RECOVERED minus BASELINE, percentage points)')
    print(q30[['country','baseline_gap_pp','without_recovered_gap_pp','change_pp']]
          .to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    mean_abs=float(q30.absolute_change_pp.mean())
    largest=q30.loc[q30.absolute_change_pp.idxmax()]
    assert len(sav_raw)==len(sav_clean)==len(csv_clean)==N_EXPECTED
    assert int(sav_clean.age_band.notna().sum())==28_148
    assert int(sav_clean.age_band_recovered.sum())==111
    scenario=make_without_recovered(sav_clean)
    assert len(scenario)==N_EXPECTED and int(scenario.age_band.isna().sum())==296
    print('\nSensitivity record counts: 28,333 people in BOTH scenarios;')
    print('  baseline 28,148 age bands known, 185 unknown;')
    print('  scenario 28,037 age bands known, 296 unknown (111 additionally unknown).')
    print('  ONLY 111 recovered age classifications set missing; weights, answers and IDs unchanged.')
    OUT.mkdir(parents=True, exist_ok=True)
    q30.to_csv(OUT/'task26_recovery_q30_country_audit.csv',index=False)
    sav_res.loc[sav_res.question.eq('ai_cncexc')].to_csv(
        OUT/'task26_recovery_q31_country_audit.csv',index=False)
    sav_groups.to_csv(OUT/'task26_recovery_age_group_audit.csv',index=False)
    sav_provenance.to_csv(OUT/'task26_recovery_country_counts.csv',index=False)
    s.to_csv(OUT/'task26_recovery_sensitivity_summary.csv',index=False)
    pd.DataFrame([{
        'task':26,
        'decision':'Use only unambiguous recovered follow-up age bands in baseline; run sensitivity with these 111 age bands unset',
        'alternative':'Use exact age/Australian reported bands only, ignoring 111 recoveries',
        'method':'Compare Pew-weighted Q30 A lot age 18-29 minus 65+ country gaps; plus all three Q31 substantive answers; DK/Refused included in denominators',
        'effect':f'111 age classifications removed from scenario (not respondents); Q30 mean absolute gap change {mean_abs:.6f} pp; largest {largest.country}: {largest.change_pp:+.6f} pp',
        'baseline_missing_age_bands':185,
        'scenario_missing_age_bands':296,
        'respondents_removed':0,
        'limitation':'Descriptive sensitivity; recovered bands help precision but cannot replace unknown exact ages',
        'verification':'Original SAV codes+labels; SAV/CSV by ID; independently reconstructed age bands; production Q30 cross-check',
        'status':'VERIFIED_WHEN_SCRIPT_PASSES',
    }]).to_csv(OUT/'task26_decision_log_entry.csv',index=False)
    pd.DataFrame([{
        'task':26,
        'ai_contribution':'Generated and tested label-driven Q30/Q31 sensitivity script and independent audits',
        'verification':'AI self-tests and CSV-only diagnostic; student runs original SAV and CSV checks',
        'ai_errors_identified':'None in delivered Task 26 script during testing',
        'source':'Pew SAV, CSV and independently validated follow-up recovery logic (Task 14/16)',
    }]).to_csv(OUT/'task26_ai_use_log_entry.csv',index=False)
    (OUT/'task26_validation_note.txt').write_text(
        'TASK 26: RECOVERED AGE-BAND SENSITIVITY\n'
        'Baseline includes all 111 respondents whose age bands were recoverable '
        'from country-specific age follow-up intervals. Scenario only changes '
        'these 111 respondent age_band values to missing. No survey person, '
        'response, survey weight or ID is deleted or changed. Original SAV labels '
        'define Q30 and Q31 codebooks. Weighted gap is percent of target response '
        'among 18-29 minus percent among 65+, within country. Both scenarios '
        'retain DK/Refused in the age-group denominator. '
        f'Baseline 28,148 known age bands, 185 missing; alternative 28,037 known, 296 missing. '
        f'Q30 mean absolute change across 24 countries: {mean_abs:.6f} percentage points. '
        f'Largest: {largest.country} {largest.change_pp:+.6f} pp. '
        'Changes are descriptive and not design-corrected inference. '
        'Original inputs unchanged; SAV and CSV independently checked by ID.\n',
        encoding='utf-8')
    print('\nSaved: task26_recovery_q30_country_audit.csv, task26_recovery_q31_country_audit.csv,')
    print('       task26_recovery_age_group_audit.csv, task26_recovery_country_counts.csv,')
    print('       task26_recovery_sensitivity_summary.csv, task26_decision_log_entry.csv,')
    print('       task26_ai_use_log_entry.csv, task26_validation_note.txt')
    print('TASK 26 PASSED: removing 111 recovered age classifications (zero respondents)')
    print(f'                changes Q30 country age gaps by {mean_abs:.6f} pp absolute mean;')
    print('                all 96 Q30/Q31 comparisons match SAV/CSV; no rows changed.')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--self-test',action='store_true')
    group.add_argument('--csv-diagnostic',action='store_true')
    arg=ap.parse_args()
    if arg.self_test:
        self_test()
    elif arg.csv_diagnostic:
        diagnostic()
    else:
        main()
