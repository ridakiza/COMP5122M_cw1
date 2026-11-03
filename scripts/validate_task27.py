"""COMP5122M P1 Task 27: Australian 'Other' gender sensitivity.

From the repository root:
    python scripts/validate_task27.py --self-test
    python scripts/validate_task27.py

Optional CSV-only preliminary diagnostic (NOT original SAV verification):
    python scripts/validate_task27.py --csv-diagnostic

Baseline retains the 21 Australian 'Other' gender responses. The alternative
sets ONLY gender_clean and gender_label to missing for those 21 respondents.
No respondent is dropped, no response, age, ID or weight changes. Thus the
age-only Q30/Q31 gaps are mathematically identical (0 pp change). Analyses
restricted to respondents with RECORDED GENDER do change: this is separately
audited, and is not confused with dropping respondents from the dataset.

Uses original SAV labels as the authority for gender and Q30/Q31 meanings.
SAV/CSV agreement is checked by respondent ID; gender cleaning is separately
reconstructed with the Task 17 validator. Does NOT overwrite pew_clean.csv.
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
    from validate_task17 import clean_gender, codebook_mapping
    from validate_task21 import assert_same_by_id, check_independent
    from validate_task25 import build_codebooks, numeric, norm_band
except ModuleNotFoundError as exc:
    raise SystemExit('Required files: src/cleaning.py, validate_task17.py, '
                     f'validate_task21.py, validate_task25.py. Import error: {exc}') from exc

SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
SNAPSHOT = ROOT / 'data/clean/pew_clean.csv'
OUT = ROOT / 'data/clean'
EXPECTED_N = 28_333
GAP_BANDS = ('18-29', '65+')
EXPECTED = {'Male': 15_107, 'Female': 13_203, 'Other': 21, 'Missing': 2}
Q30_AU_GENDER_COMPLETE_CASE_CHANGE = -1.062469938182872  # independently checked, ASSERTION only


def display_gender_counts(frame: pd.DataFrame) -> dict[str, int]:
    vals = numeric(frame.gender_clean)
    assert vals.dropna().isin([1, 2, 3]).all(), 'Unexpected cleaned gender category'
    return {
        'Male': int(vals.eq(1).sum()),
        'Female': int(vals.eq(2).sum()),
        'Other': int(vals.eq(3).sum()),
        'Missing': int(vals.isna().sum()),
    }


def make_other_missing(base: pd.DataFrame, *, strict: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Change exactly the 21 'Other' gender fields; assert zero rows deleted."""
    required = {'id', 'country_name', 'gender_clean', 'gender_label', 'age_band', 'weight',
                'ai_heard', 'ai_cncexc'}
    assert required.issubset(base.columns), f'Missing production columns: {required-set(base.columns)}'
    assert base.id.notna().all() and base.id.is_unique, 'Missing or duplicated respondent ID'
    g = numeric(base.gender_clean)
    other = g.eq(3).fillna(False)
    labels = base.gender_label.astype('string')
    assert labels.loc[other].eq('Other').all(), 'Code 3 not labelled Other'
    assert labels.loc[g.eq(1).fillna(False)].eq('Male').all()
    assert labels.loc[g.eq(2).fillna(False)].eq('Female').all()
    assert labels.loc[g.isna()].isna().all()
    assert base.loc[other, 'country_name'].eq('Australia').all(), (
        'Other gender recorded outside Australia')
    if strict:
        assert len(base) == EXPECTED_N
        assert display_gender_counts(base) == EXPECTED, 'Unexpected baseline gender totals'
        assert int(other.sum()) == 21
    alternative = base.copy(deep=True)
    alternative.loc[other, ['gender_clean', 'gender_label']] = np.nan
    assert len(base) == len(alternative) and alternative.id.equals(base.id)
    # Compare *every* column other than the TWO intentionally recoded ones.
    pd.testing.assert_frame_equal(
        base.drop(columns=['gender_clean', 'gender_label']),
        alternative.drop(columns=['gender_clean', 'gender_label']),
        check_exact=True,
    )
    assert np.isclose(
        numeric(alternative.gender_clean).to_numpy(dtype=float),
        g.mask(other).to_numpy(dtype=float),
        rtol=0, atol=0, equal_nan=True).all(), 'Alternative gender values wrong'
    assert alternative.gender_label.astype('string').fillna('<NA>').equals(
        labels.mask(other).fillna('<NA>')), 'Alternative gender labels wrong'
    assert int(alternative.gender_clean.isna().sum()) == int(g.isna().sum()) + int(other.sum())
    assert int(alternative.gender_clean.eq(3).sum()) == 0
    assert display_gender_counts(alternative)['Male'] == display_gender_counts(base)['Male']
    assert display_gender_counts(alternative)['Female'] == display_gender_counts(base)['Female']
    return alternative, other


def verify_independent_gender(raw: pd.DataFrame, clean: pd.DataFrame, meta: object, *, source: str) -> None:
    """Cross-check production gender with separate Task 17 country-routed merge."""
    independent = clean_gender(raw[['id', 'country', 'gender', 'gender_australia']], meta, strict=True)
    a = clean.set_index('id').sort_index()
    b = independent.set_index('id').sort_index()
    assert a.index.equals(b.index)
    assert np.isclose(numeric(a.gender_clean).to_numpy(dtype=float),
                      numeric(b.gender_clean).to_numpy(dtype=float),
                      rtol=0, atol=0, equal_nan=True).all(), (
        f'{source}: production gender differs from separately reconstructed gender')
    assert a.gender_label.astype('string').fillna('<NA>').equals(
        b.gender_label.astype('string').fillna('<NA>')), (
        f'{source}: independent gender labels differ')
    print(f'PASS: {source} production vs independent Task 17 gender mapping — {len(a):,} people')


def calculate_gaps(raw: pd.DataFrame, base: pd.DataFrame,
                   alternative: pd.DataFrame, names: dict[int, str],
                   specs: dict, *, strict: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Independent from production awareness_gap; original Q30/Q31 answers included.

    Scopes:
      'age_only_all_respondents': no gender filter, so both scenarios equal.
      'gender_recorded_only': restrict to respondents with known gender; only
          Australian Other can leave this ANALYTICAL subgroup (not the dataset).
    """
    assert len(raw) == len(base) == len(alternative)
    assert raw.id.notna().all() and raw.id.is_unique and base.id.is_unique
    assert np.array_equal(numeric(raw.id).to_numpy(dtype=np.int64),
                          numeric(base.id).to_numpy(dtype=np.int64))
    assert base.id.equals(alternative.id)
    raw_country = numeric(raw.country)
    source_names = raw_country.astype(int).map(names).astype('string')
    assert source_names.equals(base.country_name.astype('string'))
    weights = numeric(raw.weight)
    assert weights.notna().all() and weights.gt(0).all()
    assert np.isclose(weights.to_numpy(dtype=float), numeric(base.weight).to_numpy(dtype=float),
                      atol=1e-12, rtol=0).all()
    bands = norm_band(base.age_band)
    assert bands.dropna().isin(['18-29','30-49','50-64','65+']).all()
    assert bands.fillna('<NA>').equals(norm_band(alternative.age_band).fillna('<NA>'))
    if strict:
        assert len(raw) == EXPECTED_N and len(names) == 24
        assert bands.notna().sum() == 28_148 and bands.isna().sum() == 185
    base_g = numeric(base.gender_clean)
    alt_g = numeric(alternative.gender_clean)
    frame = pd.DataFrame({
        'country': source_names.to_numpy(),
        'age_band': bands.to_numpy(),
        'weight': weights.to_numpy(float),
        'base_has_gender': base_g.notna().to_numpy(bool),
        'alternative_has_gender': alt_g.notna().to_numpy(bool),
    })
    for question, spec in specs.items():
        src = numeric(raw[question])
        assert src.notna().all() and src.mod(1).eq(0).all()
        assert src.isin(spec['valid_codes']).all(), f'Undocumented response code in {question}'
        cleaned_expected = src.mask(src.isin(spec['nonresponse']))
        actual = numeric(base[question])
        assert np.isclose(cleaned_expected.to_numpy(float), actual.to_numpy(float),
                          rtol=0, atol=0, equal_nan=True).all(), (
            f'Cleaned {question} not consistent with SAV label meanings')
        assert np.isclose(numeric(alternative[question]).to_numpy(float),
                          actual.to_numpy(float), rtol=0, atol=0, equal_nan=True).all()
        frame[question] = src.to_numpy(dtype=int)

    rows = []
    group_rows = []
    for country in sorted(names.values()):
        country_df = frame.loc[frame.country.eq(country)]
        assert len(country_df)
        for question, spec in specs.items():
            for answer, answer_label in spec['targets'].items():
                for scope in ('age_only_all_respondents', 'gender_recorded_only'):
                    gaps = {}
                    for scenario in ('baseline', 'other_to_missing'):
                        recorded_gender = 'base_has_gender' if scenario == 'baseline' else 'alternative_has_gender'
                        percentages = {}
                        for age in GAP_BANDS:
                            group_mask = country_df.age_band.eq(age)
                            if scope == 'gender_recorded_only':
                                group_mask &= country_df[recorded_gender]
                            group = country_df.loc[group_mask]
                            n = len(group)
                            assert n > 0, f'Empty {country} {age} {scope} group'
                            den = float(group.weight.sum())
                            pos = group[question].eq(answer)
                            numerator = float(group.loc[pos, 'weight'].sum())
                            pct = numerator/den*100
                            percentages[age] = pct
                            group_rows.append({
                                'country':country, 'question':question,
                                'answer_code':int(answer), 'answer_label':answer_label,
                                'scope':scope, 'scenario':scenario, 'age_band':age,
                                'group_n':n, 'nonresponse_in_denominator_n':int(group[question].isin(spec['nonresponse']).sum()),
                                'answer_n':int(pos.sum()), 'weight_denominator':den,
                                'answer_weight_numerator':numerator, 'weighted_percentage':pct,
                            })
                        gaps[scenario] = percentages['18-29'] - percentages['65+']
                    delta = gaps['other_to_missing'] - gaps['baseline']
                    rows.append({
                        'country':country, 'question':question,
                        'answer_code':int(answer), 'answer_label':answer_label,
                        'scope':scope, 'baseline_gap_pp':gaps['baseline'],
                        'other_to_missing_gap_pp':gaps['other_to_missing'],
                        'change_pp':delta, 'absolute_change_pp':abs(delta),
                    })
    gap_audit = pd.DataFrame(rows).sort_values(['scope','question','answer_code','country']).reset_index(drop=True)
    groups_audit = pd.DataFrame(group_rows).sort_values(
        ['scope','question','answer_code','country','scenario','age_band']).reset_index(drop=True)
    expected_gaps = len(names) * sum(len(s['targets']) for s in specs.values()) * 2
    assert len(gap_audit) == expected_gaps and len(groups_audit) == expected_gaps * 4
    # Age-only estimates cannot change if only a gender covariate was recoded.
    ageonly = gap_audit[gap_audit.scope.eq('age_only_all_respondents')]
    assert len(ageonly) == expected_gaps // 2 and (ageonly.absolute_change_pp < 1e-12).all(), (
        'Age-only estimate changed unexpectedly: a respondent or another variable was altered')
    # Male/Female group membership remains exactly identical, regardless of scope.
    for code in (1, 2):
        assert np.array_equal(base_g.eq(code).fillna(False).to_numpy(dtype=bool),
                              alt_g.eq(code).fillna(False).to_numpy(dtype=bool)), (
            f'Male/Female respondent membership changed for code {code}')
    if strict:
        au = gap_audit.loc[(gap_audit.country=='Australia') &
                           (gap_audit.scope=='gender_recorded_only') &
                           (gap_audit.question=='ai_heard')]
        assert len(au)==1 and np.isclose(au.change_pp.iloc[0], Q30_AU_GENDER_COMPLETE_CASE_CHANGE,
                                         rtol=0, atol=1e-7), (
            'Australia Q30 gender-complete sensitivity differs from independent CSV check')
        other_changes = gap_audit.loc[(gap_audit.scope=='gender_recorded_only') &
                                      (gap_audit.country!='Australia')]
        assert (other_changes.absolute_change_pp < 1e-12).all(), (
            'Reclassifying Australian gender unexpectedly changed another country')
    return gap_audit, groups_audit


def compare_tables(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], *, name: str) -> None:
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    assert len(a)==len(b)
    for k in keys:
        assert a[k].astype('string').equals(b[k].astype('string')), f'{name} key mismatch {k}'
    for col in a.columns:
        if col in keys:
            continue
        v1 = numeric(a[col]).to_numpy(dtype=float)
        v2 = numeric(b[col]).to_numpy(dtype=float)
        assert np.isclose(v1, v2, rtol=0, atol=1e-9, equal_nan=True).all(), (
            f'{name} value mismatch {col}')
    print(f'PASS: {name} — {len(a)} rows match')


def crosscheck_awareness_gap(base: pd.DataFrame, alternative: pd.DataFrame,
                             gaps: pd.DataFrame) -> None:
    q = gaps.loc[(gaps.question=='ai_heard') &
                 (gaps.scope=='age_only_all_respondents')].set_index('country')
    for frame, column in ((base, 'baseline_gap_pp'),
                          (alternative, 'other_to_missing_gap_pp')):
        prod = awareness_gap(frame, weights=True, keep_dk=True).set_index('country').reindex(q.index)
        assert np.isclose(q[column].to_numpy(float), prod.gap_pp.to_numpy(float),
                          rtol=0, atol=1e-9).all(), f'Production awareness_gap differs on {column}'
    # FIX (Claude): src.cleaning.awareness_gap(drop_other_gender=True) REMOVES the 21
    # 'Other' people from the calculation (x.gender_clean != 3), so it is the
    # gender-recorded scope, not the age-only scope. The original assertion (that it
    # equals the baseline) was wrong and crashed. Check it against the right scope.
    prod_alt = (awareness_gap(base, weights=True, keep_dk=True, drop_other_gender=True)
                .set_index('country').reindex(q.index))
    gr = gaps.loc[(gaps.question=='ai_heard') &
                  (gaps.scope=='gender_recorded_only')].set_index('country').reindex(q.index)
    assert np.isclose(prod_alt.gap_pp.to_numpy(float),
                      gr['other_to_missing_gap_pp'].to_numpy(float), rtol=0, atol=1e-9).all(), (
        'cleaning.py drop_other_gender does not match the gender-recorded scope')
    changed = q.index[~np.isclose(prod_alt.gap_pp.to_numpy(float),
                                  q['baseline_gap_pp'].to_numpy(float), rtol=0, atol=1e-12)]
    assert list(changed) == ['Australia'], f'Unexpected countries changed: {list(changed)}'
    print('PASS: independent Q30 age-only gaps match production awareness_gap() (unchanged)')
    print('NOTE: cleaning.py awareness_gap(drop_other_gender=True) drops the 21 Other people;')
    print('      it equals the gender-recorded result (Australia only changes), not the age-only one')


def summary_table(gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, question, answer_code, answer_label), grp in gaps.groupby(
            ['scope','question','answer_code','answer_label'], sort=True):
        biggest = grp.loc[grp.absolute_change_pp.idxmax()]
        rows.append({
            'scope':scope, 'question':question, 'answer_code':int(answer_code),
            'answer_label':answer_label, 'countries':len(grp),
            'mean_absolute_change_pp':float(grp.absolute_change_pp.mean()),
            'max_abs_change_country':biggest.country,
            'max_signed_change_pp':float(biggest.change_pp),
            'countries_changed':int(grp.absolute_change_pp.gt(1e-10).sum()),
        })
    return pd.DataFrame(rows).sort_values(['scope','question','answer_code'])


def should_fail(fn, label: str) -> None:
    try:
        fn()
    except (AssertionError, ValueError, KeyError):
        print('PASS:',label)
    else:
        raise AssertionError('Failure went undetected: '+label)


def self_test() -> None:
    meta = SimpleNamespace(variable_value_labels={
        'country':{1:'Australia',2:'Canada'},
        'gender':{1:'Male',2:'Female'},
        'gender_australia':{1:'Male',2:'Female',3:'Other',99:"Don't know/Refused"},
        'ai_heard':{1:'A lot',2:'A little',3:'Nothing at all',8:"Don't know",9:'Refused'},
        'ai_cncexc':{1:'More excited than concerned',2:'More concerned than excited',
                     3:'Equally concerned and excited',8:"Don't know",9:'Refused'},
    })
    raw_gender = pd.DataFrame({
        'id':range(1,9),
        'country':[1,1,1,1,2,2,2,2],
        'gender':[np.nan,np.nan,np.nan,np.nan,1,2,1,2],
        'gender_australia':[3,1,2,np.nan,np.nan,np.nan,np.nan,np.nan],
    })
    merged = clean_gender(raw_gender,meta,strict=False)
    merged['country_name'] = pd.Series(['Australia']*4+['Canada']*4)
    merged['age_band'] = pd.Series(['18-29','18-29','65+','65+', '18-29','18-29','65+','65+'])
    merged['weight'] = pd.Series([3.,1.,1.,1.,1.,1.,1.,1.])
    merged['ai_heard'] = pd.Series([1,2,1,8,1,2,1,9],dtype=float)
    merged['ai_cncexc'] = pd.Series([1,2,3,8,2,1,3,9],dtype=float)
    source = merged.copy(deep=True)
    for question in ('ai_heard','ai_cncexc'):
        source[question] = source[question].astype(float)
    original = merged.copy(deep=True)
    # do_clean would replace DK/refused with missing, as Task 27 must expect.
    merged.loc[merged.ai_heard.eq(8)|merged.ai_heard.eq(9),'ai_heard'] = np.nan
    merged.loc[merged.ai_cncexc.eq(8)|merged.ai_cncexc.eq(9),'ai_cncexc'] = np.nan
    alt, other = make_other_missing(merged,strict=False)
    assert int(other.sum())==1
    assert len(alt)==len(merged)==8
    assert display_gender_counts(alt)=={'Male':3,'Female':3,'Other':0,'Missing':2}
    assert merged.loc[0,'gender_clean']==3 and pd.isna(alt.loc[0,'gender_clean'])
    pd.testing.assert_frame_equal(source,original,check_exact=True)
    gap, groups = calculate_gaps(source,merged,alt,{1:'Australia',2:'Canada'},
                                 build_codebooks(meta), strict=False)
    assert len(gap)==16 and len(groups)==64  # 2 countries x 4 answers x 2 scopes
    assert gap.loc[gap.scope=='age_only_all_respondents','absolute_change_pp'].max()==0
    au = gap.loc[(gap.country=='Australia') & (gap.scope=='gender_recorded_only') &
                 (gap.question=='ai_heard')]
    assert float(au.change_pp.iloc[0])!=0, 'Gender-filtered sample should react to recoding'
    print('PASS: only Other gender is made missing; all IDs, rows, responses, ages and weights unchanged')
    print('PASS: age-only Q30/Q31 gaps unchanged; gender-complete estimate can change')
    should_fail(lambda: make_other_missing(merged.drop(columns=['gender_clean']), strict=False),
                'missing gender column detected')
    broken = merged.copy(deep=True)
    broken.loc[0,'country_name']='Canada'
    should_fail(lambda: make_other_missing(broken,strict=False),
                'Other gender outside Australia rejected')
    broken = source.copy(deep=True)
    broken.loc[0,'ai_heard']=7
    should_fail(lambda: calculate_gaps(broken,merged,alt,{1:'Australia',2:'Canada'},
                                       build_codebooks(meta),strict=False),
                'unlabelled Q30 source code rejected')
    broken = alt.drop(index=0)
    should_fail(lambda: calculate_gaps(source,merged,broken,{1:'Australia',2:'Canada'},
                                       build_codebooks(meta),strict=False),
                'incorrectly dropping an Other respondent detected')
    wrong_meta = SimpleNamespace(variable_value_labels={
        **meta.variable_value_labels,
        'gender_australia':{1:'Male',2:'Female',3:"Don't know",99:'Refused'},
    })
    should_fail(lambda: codebook_mapping(wrong_meta), 'mislabelled Other rejected by SAV metadata')
    print('TASK 27 SELF-TEST PASSED')


def diagnostic() -> None:
    assert SNAPSHOT.is_file() and CSV.is_file(), 'Missing CSV/snapshot for preliminary diagnostic'
    raw = pd.read_csv(CSV, encoding='utf-8-sig', low_memory=False)
    clean = pd.read_csv(SNAPSHOT,low_memory=False)
    dummy = SimpleNamespace(variable_value_labels={
        'gender':{1:'Male',2:'Female'},
        'gender_australia':{1:'Male',2:'Female',3:'Other',99:"Don't know/Refused"},
        'ai_heard':{1:'A lot',2:'A little',3:'Nothing at all',8:"Don't know",9:'Refused'},
        'ai_cncexc':{1:'More excited than concerned',2:'More concerned than excited',
                     3:'Equally concerned and excited',8:"Don't know",9:'Refused'},
    })
    names = {int(k):str(v) for k,v in clean[['country','country_name']].drop_duplicates().itertuples(index=False,name=None)}
    dummy.variable_value_labels['country'] = {k:v for k,v in names.items()}
    assert len(raw)==len(clean)==EXPECTED_N
    assert_same_by_id(raw,clean,['country','weight'],context='CSV raw vs cleaned snapshot')
    verify_independent_gender(raw,clean,dummy,source='CSV diagnostic')
    alt,_=make_other_missing(clean)
    gaps, groups = calculate_gaps(raw,clean,alt,names,build_codebooks(dummy))
    summary=summary_table(gaps)
    print('=== TASK 27: CSV-ONLY DIAGNOSTIC (NOT ORIGINAL SAV VERIFICATION) ===')
    print(summary.to_string(index=False, float_format=lambda x:f'{x:.6f}'))
    print('NOTE: diagnostic label values are provided explicitly as proxies; original SAV labels are checked in main mode.')


def main() -> None:
    try:
        import pyreadstat
    except ImportError as exc:
        raise SystemExit('Install pyreadstat into the active venv to validate the original SAV') from exc
    assert SAV.is_file() and CSV.is_file(), 'Place both original Pew SAV and CSV under data/raw/'
    print('=== TASK 27: AUSTRALIA OTHER GENDER — KEEP VS SET MISSING ===')
    sav_raw, meta = pyreadstat.read_sav(str(SAV),apply_value_formats=False,user_missing=True)
    csv_raw = pd.read_csv(CSV,encoding='utf-8-sig',low_memory=False)
    sav_raw.columns = sav_raw.columns.str.strip().str.lower()
    csv_raw.columns = csv_raw.columns.str.strip().str.lower()
    assert sav_raw.shape==csv_raw.shape==(EXPECTED_N,483)
    name_map = country_names(meta)
    mappings, gender_labels = codebook_mapping(meta)
    assert mappings['gender_australia'].get(3)==3 and mappings['gender_australia'].get(99) is None
    print('Original SAV Australian gender labels:',gender_labels['gender_australia'])
    specs = build_codebooks(meta)
    fields=['country','weight','gender','gender_australia','age','age_australia','ai_heard','ai_cncexc']
    assert_same_by_id(sav_raw,csv_raw,fields,context='Original SAV vs CSV')
    sav_before=sav_raw.copy(deep=True)
    csv_before=csv_raw.copy(deep=True)
    sav_clean=do_clean(sav_raw,meta,log=[])
    csv_clean=do_clean(csv_raw,meta,log=[])
    pd.testing.assert_frame_equal(sav_raw,sav_before,check_exact=True)
    pd.testing.assert_frame_equal(csv_raw,csv_before,check_exact=True)
    print('PASS: original SAV/CSV unchanged by production cleaning')
    assert_same_by_id(sav_clean,csv_clean,
                      ['country','weight','age_band','age_band_recovered','gender_clean',
                       'ai_heard','ai_cncexc'],context='Cleaned SAV vs CSV')
    verify_independent_gender(sav_raw,sav_clean,meta,source='SAV')
    verify_independent_gender(csv_raw,csv_clean,meta,source='CSV')
    check_independent(sav_raw,sav_clean,meta,name='SAV')
    check_independent(csv_raw,csv_clean,meta,name='CSV')
    sav_alt,sav_other = make_other_missing(sav_clean)
    csv_alt,csv_other = make_other_missing(csv_clean)
    assert_same_by_id(sav_alt,csv_alt,
                      ['country','weight','age_band','age_band_recovered','gender_clean',
                       'ai_heard','ai_cncexc'],context='Alternative SAV vs CSV')
    assert int(sav_other.sum())==int(csv_other.sum())==21
    sav_gaps,sav_groups=calculate_gaps(sav_raw,sav_clean,sav_alt,name_map,specs)
    csv_gaps,csv_groups=calculate_gaps(csv_raw,csv_clean,csv_alt,name_map,specs)
    compare_tables(sav_gaps,csv_gaps,
                   ['country','question','answer_code','answer_label','scope'],name='All 192 country-response-scope comparisons SAV/CSV')
    compare_tables(sav_groups,csv_groups,
                   ['country','question','answer_code','answer_label','scope','scenario','age_band'],
                   name='All 768 group-level counts and weighted numerators/denominators SAV/CSV')
    crosscheck_awareness_gap(sav_clean,sav_alt,sav_gaps)
    crosscheck_awareness_gap(csv_clean,csv_alt,csv_gaps)
    rows=[]
    initial=display_gender_counts(sav_clean)
    changed=display_gender_counts(sav_alt)
    for gender in ('Male','Female','Other','Missing'):
        rows.append({'gender':gender,'keep_other':initial[gender],
                     'other_to_missing':changed[gender],
                     'difference':changed[gender]-initial[gender]})
    genders=pd.DataFrame(rows)
    assert initial==EXPECTED and changed=={'Male':15107,'Female':13203,'Other':0,'Missing':23}
    assert genders.keep_other.sum()==genders.other_to_missing.sum()==EXPECTED_N
    print('\nGender distribution (no respondents dropped):')
    print(genders.to_string(index=False))
    youngold = sav_clean.loc[sav_clean.country_name.eq('Australia')].copy()
    youngold['norm_age_band'] = norm_band(youngold.age_band)
    band_counts=[]
    for band in ('18-29','30-49','50-64','65+','Missing'):
        in_band = youngold.norm_age_band.isna() if band=='Missing' else youngold.norm_age_band.eq(band)
        band_counts.append({'age_band':band,'australian_other_n':int((in_band & youngold.gender_clean.eq(3)).sum())})
    age_other=pd.DataFrame(band_counts)
    assert age_other.australian_other_n.sum()==21
    print('\nAustralian 21 Other answers by age band:')
    print(age_other.to_string(index=False))
    summary=summary_table(sav_gaps)
    print('\nGap sensitivity (24-country mean absolute change, percentage points):')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    au=sav_gaps.loc[(sav_gaps.country=='Australia') & (sav_gaps.scope=='gender_recorded_only')]
    print('\nAustralia GENDER-RECORDED subset (NOT the full-sample age-only gap):')
    print(au[['question','answer_label','baseline_gap_pp','other_to_missing_gap_pp','change_pp']]
          .to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    assert int(sav_clean.age_band.isna().sum()) == 185
    assert int(sav_alt.age_band.isna().sum()) == 185
    assert sav_clean.id.equals(sav_alt.id) and len(sav_alt)==EXPECTED_N
    OUT.mkdir(parents=True,exist_ok=True)
    genders.to_csv(OUT/'task27_gender_other_counts.csv',index=False)
    age_other.to_csv(OUT/'task27_australia_other_age_audit.csv',index=False)
    sav_gaps.loc[sav_gaps.scope=='age_only_all_respondents'].to_csv(
        OUT/'task27_age_only_country_audit.csv',index=False)
    sav_gaps.loc[sav_gaps.scope=='gender_recorded_only'].to_csv(
        OUT/'task27_gender_complete_country_audit.csv',index=False)
    sav_groups.to_csv(OUT/'task27_group_denominator_audit.csv',index=False)
    summary.to_csv(OUT/'task27_sensitivity_summary.csv',index=False)
    pd.DataFrame([{
        'task':27,
        'decision':'Keep 21 original Australian Other genders as a legitimate third category',
        'alternative':'Set ONLY gender_clean and gender_label missing for the 21 Other records (no row filtering/deletion)',
        'rationale':'SAV explicitly labels Other as valid, distinct from labelled nonresponse code 99',
        'effect':'Baseline gender known 28,331, missing 2; alternative known 28,310, missing 23; same 28,333 rows',
        'age_only_q30_q31_gap_change_pp':0,
        'australia_gender_recorded_only_q30_age_gap_change_pp':Q30_AU_GENDER_COMPLETE_CASE_CHANGE,
        'scope_caveat':'Age-only estimates are unchanged; analyses requiring recorded gender can change',
        'verification':'Original SAV metadata, original SAV/CSV by ID, independent gender and age mappings, production Q30 cross-check',
        'status':'VERIFIED_WHEN_SCRIPT_PASSES',
    }]).to_csv(OUT/'task27_decision_log_entry.csv',index=False)
    pd.DataFrame([{
        'task':27,
        'ai_contribution':'Designed and tested reversible gender-only recode and age-only versus gender-complete sensitivity audits',
        'human_verification':'Original SAV and CSV executed by student; all records and codebook labels checked',
        'independent_checks':'Task 17 gender mapping, Task 14/16 age mapping, independent original response weighted calculations',
        'ai_errors':'Previous AI starter erroneously removed 21 people; corrected to recode gender values only (documented separately)',
    }]).to_csv(OUT/'task27_ai_use_log_entry.csv',index=False)
    pd.DataFrame([{
        'task':27,
        'original_ai_error':'Earlier draft of sensitivity removed 21 Other respondents instead of retaining records and marking only gender missing',
        'how_detected':'Review of earlier AI-generated cleaning/sensitivity logic; comparison with stated record-retention requirement',
        'correction':'Only gender_clean and gender_label are set missing for those 21 in a COPY; assert 28,333 rows and unchanged IDs, weights, ages, responses and all other fields',
        'validation':'Task27 compares original and alternative frame across every unchanged column and runs SAV/CSV respondent-level audits',
        'effect_on_data':'No source data modified; actual alternative has 23 missing gender, not fewer rows',
        'status':'RESOLVED_IN_VALIDATOR; pending_student_SAV_execution',
    },{
        'task':27,
        'original_ai_error':'Initial Task27 synthetic test used full-survey expected table length and dtype-sensitive pandas Series.equals() for Boolean membership',
        'how_detected':'Generated script failed its first two local synthetic self-test runs, before delivery',
        'correction':'Derive expected comparison-table length from number of tested countries/questions, and compare equivalent Boolean masks by values rather than Series dtype',
        'validation':'Retested full synthetic suite, CSV diagnostic, and entire validation main path using original CSV with dictionary-derived metadata; all pass',
        'effect_on_data':'No survey data affected; the defect was in the draft validator assertions',
        'status':'RESOLVED_BEFORE_DELIVERY; pending_student_SAV_execution',
    }]).to_csv(OUT/'task27_ai_discrepancy_entry.csv',index=False)
    (OUT/'task27_validation_note.txt').write_text(
        'TASK 27: GENDER OTHER SENSITIVITY\n'
        'Retain Australian gender code labelled Other (21 responses). Scenario '
        'changes only these 21 gender_clean and gender_label values to missing, '
        'NOT 21 respondent records. All IDs, weights, original survey answers, '
        'and age bands remain identical. Baseline gender Male 15,107, Female 13,203, '
        'Other 21, Missing 2; scenario Male 15,107, Female 13,203, Other 0, Missing 23. '
        'All 96 weighted country-age-only Q30/Q31 gaps are exactly unchanged: '
        'the measure does not condition on gender. However, an analysis restricted '
        'to people with recorded gender is NOT invariant: Australia Q30 age gap '
        f'changes {Q30_AU_GENDER_COMPLETE_CASE_CHANGE:+.6f} pp, since 11 of the 21 '
        'Other respondents are aged 18-29. The main analysis retains Other; '
        'the complete-case restriction is only a sensitivity demonstration. '
        'The original survey data remain unchanged. Prior AI error that deleted '
        'Other respondents is recorded and corrected. SAV and CSV are checked '
        'independently by ID and labels. No full-survey variance claims.\n',encoding='utf-8')
    print('\nSaved: task27_gender_other_counts.csv, task27_australia_other_age_audit.csv,')
    print('       task27_age_only_country_audit.csv, task27_gender_complete_country_audit.csv,')
    print('       task27_group_denominator_audit.csv, task27_sensitivity_summary.csv,')
    print('       task27_decision_log_entry.csv, task27_ai_use_log_entry.csv,')
    print('       task27_ai_discrepancy_entry.csv, task27_validation_note.txt')
    print('TASK 27 PASSED: 21 Other gender values changed to missing IN A COPY, zero rows removed;')
    print('                96/96 age-only Q30/Q31 age gaps change by 0.000000 pp;')
    print('                Australia gender-complete Q30 gap changes by -1.062470 pp;')
    print('                all SAV/CSV and independent age/gender checks agree.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    options=parser.add_mutually_exclusive_group()
    options.add_argument('--self-test',action='store_true')
    options.add_argument('--csv-diagnostic',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
    elif args.csv_diagnostic:
        diagnostic()
    else:
        main()