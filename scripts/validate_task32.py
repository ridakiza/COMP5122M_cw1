"""COMP5122M Coursework 1, Task 32 — exploratory 483-column association screen.

Authoritative original SAV + original CSV:
    python scripts/validate_task32.py --self-test
    python scripts/validate_task32.py

Local CSV/dictionary diagnostic (NOT authoritative SAV validation):
    python scripts/validate_task32.py --csv-diagnostic --labels-json PATH_TO_JSON

Use EXACTLY the same unweighted, uncorrected Cramer's V against Q30's three
substantive categories for internet frequency and the FINAL four-level age_band.
The other 483 original columns are examined and marked either 'tested' or with
an explicit exclusion reason (such as date, identifier, >25 levels or <100 cases).
All per-variable nonresponses come from the original SAV labels: especially
Q32's 8=Not sure stays valid, internetfreq 6=non-user stays valid.

This is an exploratory, unadjusted, multiple-comparison screen; it neither
estimates causal effects nor incorporates the complex survey design. No source
or final cleaned data is modified and only AGGREGATE output is saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.cleaning import do_clean, missing_codes, screen_all_columns

SAV = ROOT/'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = ROOT/'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
OUT = ROOT/'data/clean'
EXPORT = OUT/'pew_clean_p1_provisional.csv'
MANIFEST = OUT/'task29_provisional_export_manifest.csv'
N = 28_333
RAW_VARIABLES = 483
DATE_FIELDS = frozenset(('qdate_s','qdate_e','intdate_aus'))
EXCLUDE = frozenset(('id','weight','ai_heard','survey'))
MAX_LEVELS = 25
MIN_COMPLETE = 100
Q30_COUNTS = {1:9407, 2:12878, 3:5240, 8:785, 9:23}
EXPECTED_FOCUS = {
    'internetfreq': (25_625, 0.2596166610791511, 6),
    'age_band': (27_340, 0.08510813443855163, 4),
}


def numeric(series: pd.Series, *, label: str) -> pd.Series:
    """Parse numeric source data strictly. Whitespace is structural missing."""
    if pd.api.types.is_numeric_dtype(series):
        result = pd.to_numeric(series, errors='raise')
    else:
        st = series.astype('string').str.strip()
        st = st.mask(st.eq(''))
        result = pd.to_numeric(st, errors='coerce')
        unexpected = st.notna() & result.isna()
        if unexpected.any():
            raise ValueError(f'{label}: {int(unexpected.sum())} nonnumeric values '
                             f'(sample={st.loc[unexpected].head(3).tolist()})')
    if not np.isfinite(result.dropna().to_numpy(dtype=float)).all():
        raise AssertionError(f'{label}: infinite or invalid numbers')
    return result


def missing_masked(s: pd.Series, meta: object, col: str) -> pd.Series:
    n = numeric(s, label=col)
    return n.mask(n.isin(missing_codes(meta, col)))


def q30_target(raw: pd.DataFrame, meta: object) -> pd.Series:
    labels = {int(float(k)): str(v).lower().replace('’', "'")
              for k, v in (next((d for key, d in meta.variable_value_labels.items()
                                 if str(key).strip().lower() == 'ai_heard'), {})).items()}
    assert len(labels) == 5 and sum(v.strip() == 'a lot' for v in labels.values()) == 1, (
        'Q30 SAV labels missing or unexpected')
    assert {k for k, v in labels.items() if "don't know" in v or 'refused' in v} == {8, 9}, (
        'Q30 DK/refusal codes changed; recheck the original SAV labels')
    y = missing_masked(raw.ai_heard, meta, 'ai_heard')
    assert set(y.dropna().unique()) == {1, 2, 3}, 'Q30 valid categories are not 1–3'
    return y


def cramers_v(x: pd.Series, y: pd.Series, *, min_n: int = MIN_COMPLETE,
              max_levels: int = MAX_LEVELS) -> dict:
    """Unweighted Cramer's V, without Yates correction or bias adjustment.

    Pairwise complete cases: exclude *labelled nonresponses* and structural
    blanks in EITHER variable; do not assume every raw code 8/9 is missing.
    """
    assert len(x) == len(y), 'Predictor and Q30 respondent counts differ'
    assert x.index.equals(y.index), 'Predictor and target are not aligned'
    levels = int(x.dropna().nunique())
    row = {'levels_all_available': levels,
           'n_joint_complete': int((x.notna() & y.notna()).sum()),
           'cramers_v': np.nan, 'n_x_levels_in_table': 0,
           'n_y_levels_in_table': 0,
           'min_expected_cell': np.nan, 'share_expected_cells_below_5': np.nan,
           'status': 'tested'}
    if levels < 2:
        row['status'] = 'skip_single_level_or_all_missing'
        return row
    if levels > max_levels:
        row['status'] = 'skip_high_cardinality_over_25_levels'
        return row
    if row['n_joint_complete'] < min_n:
        row['status'] = 'skip_fewer_than_100_complete_cases'
        return row
    tab = pd.crosstab(x, y, dropna=True)
    row['n_x_levels_in_table'] = tab.shape[0]
    row['n_y_levels_in_table'] = tab.shape[1]
    if min(tab.shape) < 2:
        row['status'] = 'skip_degenerate_complete_case_table'
        return row
    chi2, _, _, expected = chi2_contingency(tab.to_numpy(), correction=False)
    n = int(tab.to_numpy().sum())
    assert n == row['n_joint_complete']
    v = np.sqrt((chi2/n)/min(tab.shape[0]-1, tab.shape[1]-1))
    assert np.isfinite(v) and -1e-10 <= v <= 1+1e-8, f'Invalid Cramer V: {v}'
    row.update({'cramers_v': float(v), 'min_expected_cell': float(expected.min()),
                'share_expected_cells_below_5': float((expected < 5).mean())})
    return row


def scan_483(raw: pd.DataFrame, meta: object, *, strict: bool = True) -> pd.DataFrame:
    if strict:
        assert raw.shape == (N, RAW_VARIABLES), f'Expected original 28,333x483, got {raw.shape}'
    assert not raw.columns.duplicated().any(), 'Duplicate original column names'
    assert 'id' in raw and raw.id.notna().all() and raw.id.is_unique
    y = q30_target(raw, meta)
    rows = []
    for col in raw.columns:
        # Excluded columns must still have accurate source coverage in the audit.
        present_n = int(raw[col].astype('string').str.strip().ne('').fillna(False).sum())
        base = {'variable': col, 'source': 'original_survey_column',
                'original_nonblank_n': present_n, 'labelled_nonresponse_n': 0,
                'n_after_labelled_missing': present_n, 'n_complete_y': int(y.notna().sum()),
                'n_total_respondents': len(raw),
                'labelled_nonresponse_codes': '',
                'codebook_warning': ('French income source has five undocumented 98s; '
                                     'not silently recoded' if col == 'd_income2_france' else '')}
        if col in DATE_FIELDS:
            rows.append({**base, 'status':'skip_date_field', 'levels_all_available':np.nan,
                         'n_joint_complete':np.nan, 'n_x_levels_in_table':np.nan,
                         'n_y_levels_in_table':np.nan, 'cramers_v':np.nan,
                         'min_expected_cell':np.nan, 'share_expected_cells_below_5':np.nan})
            continue
        if col in EXCLUDE:
            rows.append({**base, 'status':'skip_identifier_weight_or_outcome',
                         'levels_all_available':np.nan, 'n_joint_complete':np.nan,
                         'n_x_levels_in_table':np.nan,'n_y_levels_in_table':np.nan,
                         'cramers_v':np.nan,'min_expected_cell':np.nan,
                         'share_expected_cells_below_5':np.nan})
            continue
        raw_x = numeric(raw[col], label=col)
        bad = missing_codes(meta, col)
        x = raw_x.mask(raw_x.isin(bad))
        stats = cramers_v(x, y)
        rows.append({**base, **stats,
                     'original_nonblank_n':int(raw_x.notna().sum()),
                     'labelled_nonresponse_n':int(raw_x.isin(bad).sum()),
                     'n_after_labelled_missing':int(x.notna().sum()),
                     'labelled_nonresponse_codes':'|'.join(f'{v:g}' for v in sorted(bad))})
    result = pd.DataFrame(rows)
    assert len(result) == len(raw.columns) and result.variable.is_unique
    assert result.variable.tolist() == list(raw.columns)
    assert (result.status.eq('tested') == result.cramers_v.notna()).all(), (
        'Every tested predictor must have exactly one statistic; exclusions must not')
    return result


def unified_age_record(raw: pd.DataFrame, cleaned: pd.DataFrame,
                       meta: object) -> dict:
    assert len(raw) == len(cleaned) and raw.id.equals(cleaned.id)
    x = cleaned.age_band.astype('string').str.strip().str.replace('–','-', regex=False)
    x = x.str.replace('—', '-', regex=False)
    assert set(x.dropna().unique()) == {'18-29', '30-49', '50-64', '65+'}
    assert int(x.notna().sum()) == 28_148 and int(x.isna().sum()) == 185
    y = q30_target(raw, meta)
    row = cramers_v(x, y)
    assert row['status'] == 'tested'
    return {'variable':'age_band','source':'derived_P1_age_band_not_in_original_483',
            'original_nonblank_n':int(x.notna().sum()),'labelled_nonresponse_n':0,
            'n_after_labelled_missing':int(x.notna().sum()),
            'n_complete_y':int(y.notna().sum()),'n_total_respondents':len(x),
            'labelled_nonresponse_codes':'', 'codebook_warning':'Includes 111 verified recoveries; '
            '185 remain missing; not identical to the original continuous age field', **row}


def verify_reference(scan: pd.DataFrame, age: dict) -> None:
    assert len(scan) == 483 and scan.variable.is_unique
    k = scan.set_index('variable')
    a = k.loc['internetfreq']
    for name, rec in [('internetfreq', a), ('age_band', age)]:
        n, v, levels = EXPECTED_FOCUS[name]
        assert rec['status'] == 'tested', f'{name}: not tested'
        assert int(rec['n_joint_complete']) == n, f'{name}: unexpected complete-case N'
        assert int(rec['n_x_levels_in_table']) == levels, f'{name}: unexpected levels'
        assert abs(float(rec['cramers_v'])-v) < 1e-9, (
            f'{name}: unexpected association, found {rec["cramers_v"]}; '
            'check denominator, source data, missing labels and estimator')
    assert int(a.labelled_nonresponse_n) == 94
    print('PASS: same unweighted Cramer V statistic used for internetfreq and final age_band')
    print('PASS: internetfreq n=25,625, V=0.259616661; age_band n=27,340, V=0.085108134')
    print('NOTE: age V~0.22 from the checklist is NOT supported by this same-metric comparison')


def compare_scan(a: pd.DataFrame, b: pd.DataFrame) -> None:
    assert a.variable.tolist() == b.variable.tolist(), '483 original variables differ'
    assert a.status.tolist() == b.status.tolist(), 'Original SAV/CSV scan eligibility differs'
    for name in ('original_nonblank_n','labelled_nonresponse_n',
                 'n_after_labelled_missing','n_joint_complete',
                 'levels_all_available','n_x_levels_in_table','n_y_levels_in_table'):
        aa = pd.to_numeric(a[name],errors='coerce').to_numpy(dtype=float)
        bb = pd.to_numeric(b[name],errors='coerce').to_numpy(dtype=float)
        assert np.allclose(aa,bb,rtol=0,atol=0,equal_nan=True), f'SAV/CSV differ on {name}'
    for name in ('cramers_v','min_expected_cell','share_expected_cells_below_5'):
        aa = a[name].to_numpy(dtype=float)
        bb = b[name].to_numpy(dtype=float)
        assert np.allclose(aa,bb,rtol=0,atol=1e-11,equal_nan=True), (
            f'SAV/CSV differ on association metric {name}')
    print(f'PASS: SAV/CSV agree on status, counts and Cramer V for ALL {len(a)} raw columns')


def independent_existing_function(raw: pd.DataFrame, meta: object,
                                  table: pd.DataFrame) -> None:
    """Compare tested raw values against separate pre-existing coursework implementation."""
    prior = screen_all_columns(raw, meta)
    a = table.loc[table.status.eq('tested'),['variable','n_joint_complete','cramers_v']]
    b = prior[['variable','n','cramers_v']].rename(columns={'n':'n_joint_complete'})
    compare = a.merge(b,on='variable',how='outer',indicator=True,suffixes=('_new','_existing'))
    assert compare['_merge'].eq('both').all(), (
        'New scan differs from existing src.cleaning.screen_all_columns eligibility')
    assert np.array_equal(compare.n_joint_complete_new.to_numpy(),
                          compare.n_joint_complete_existing.to_numpy())
    assert np.allclose(compare.cramers_v_new,compare.cramers_v_existing,atol=1e-12,rtol=0)
    print(f'PASS: {len(compare)} tested variables also agree with independent '
          'src.cleaning.screen_all_columns() implementation')


def compare_critical_by_id(sav: pd.DataFrame, csv: pd.DataFrame,
                           sav_clean: pd.DataFrame, csv_clean: pd.DataFrame) -> None:
    def aligned(df):
        idx = numeric(df.id,label='id').astype('int64')
        assert idx.is_unique and idx.notna().all() and len(df)==N
        copy = df.copy(deep=False)
        copy.index = idx
        return copy.sort_index()
    a,b = aligned(sav),aligned(csv)
    assert a.index.equals(b.index), 'Respondents differ between SAV and CSV'
    for field in ('country','weight','ai_heard','internetfreq'):
        x = numeric(a[field],label=f'SAV {field}').to_numpy(dtype=float,na_value=np.nan)
        y = numeric(b[field],label=f'CSV {field}').to_numpy(dtype=float,na_value=np.nan)
        assert np.isclose(x,y,atol=(1e-10 if field=='weight' else 0),rtol=0,equal_nan=True).all(), (
            f'Original SAV/CSV respondent mismatch: {field}')
    a,b = aligned(sav_clean),aligned(csv_clean)
    assert a.index.equals(b.index)
    for field in ('age_band','age_band_source'):
        x = a[field].astype('string').str.replace('–','-',regex=False).fillna('<missing>')
        y = b[field].astype('string').str.replace('–','-',regex=False).fillna('<missing>')
        assert x.equals(y), f'SAV/CSV derived {field} differs for some respondents'
    print('PASS: 28,333 SAV/CSV IDs and critical raw/derived columns match person by person')


def load_sources(diagnostic: bool, labels_json: Path|None):
    if not CSV.is_file():
        raise FileNotFoundError('Original CSV not found in data/raw')
    csv = pd.read_csv(CSV,low_memory=False,encoding='utf-8-sig')
    csv.columns = csv.columns.str.lower().str.strip()
    if diagnostic:
        if labels_json is None or not labels_json.is_file():
            raise FileNotFoundError('CSV diagnostic needs --labels-json from the supplied data dictionary')
        labels = json.loads(labels_json.read_text(encoding='utf-8'))
        meta = SimpleNamespace(variable_value_labels={
            col:{float(code):val for code,val in mapping.items()}
            for col,mapping in labels.items()})
        return csv.copy(deep=True),csv,meta
    try:
        import pyreadstat
    except ImportError as exc:
        raise RuntimeError('Original SAV validation requires pyreadstat; '
                           'run `python -m pip install pyreadstat`') from exc
    if not SAV.is_file():
        raise FileNotFoundError('Original SAV not found in data/raw')
    sav,meta = pyreadstat.read_sav(str(SAV),apply_value_formats=False,user_missing=True)
    sav.columns = sav.columns.str.lower().str.strip()
    return sav,csv,meta


def sha256(path:Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def optional_export_integrity(clean: pd.DataFrame) -> None:
    """Check against Task29 export when present, but do not create a dataset."""
    if not EXPORT.exists() or not MANIFEST.exists():
        print('NOTE: no Task 29 manifest/export in this environment; skipped export check')
        return
    manifest = pd.read_csv(MANIFEST)
    assert len(manifest)==1 and manifest.sha256.iloc[0]==sha256(EXPORT), (
        'Task 29 export digest differs from its manifest')
    prior=pd.read_csv(EXPORT,usecols=['id','age_band'],low_memory=False)
    a=prior.set_index('id').sort_index().age_band.astype('string').str.replace('–','-',regex=False)
    b=clean.set_index('id').sort_index().age_band.astype('string').str.replace('–','-',regex=False)
    assert a.fillna('<NA>').equals(b.fillna('<NA>')), 'Task29 export age bands differ'
    print('PASS: Task 29 provisional export checksum and respondent-level age bands verified')


def write_outputs(scan:pd.DataFrame, age:dict, *, diagnostic:bool) -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    prefix='task32_csv_diagnostic_' if diagnostic else 'task32_'
    all_rows=pd.concat([scan,pd.DataFrame([age])],ignore_index=True)
    ranking=(all_rows.loc[all_rows.status.eq('tested')]
             .sort_values(['cramers_v','n_joint_complete'],ascending=[False,False])
             .reset_index(drop=True))
    ranking.insert(0,'rank',np.arange(1,len(ranking)+1))
    focus=all_rows.loc[all_rows.variable.isin(['internetfreq','age_band'])].copy()
    focus['comparison_note']=[('Age is the final four-band derived variable; not the raw continuous age '
                              'and not the claimed 0.22') if v=='age_band'
                              else ("Internet use excludes Australia's structurally unasked records, "
                                    'but keeps code 6 non-users')
                              for v in focus.variable]
    scan.to_csv(OUT/f'{prefix}full_483_variable_scan.csv',index=False)
    ranking.to_csv(OUT/f'{prefix}associations_ranked.csv',index=False)
    focus.to_csv(OUT/f'{prefix}key_comparisons.csv',index=False)
    flag=bool(scan.loc[scan.variable.eq('d_income2_france'),'codebook_warning'].str.len().gt(0).all())
    assert flag
    note=(
        'TASK 32 — EXPLORATORY 483-VARIABLE ASSOCIATION SCREEN\n'
        f"Source: {'CSV + dictionary labels, diagnostic ONLY' if diagnostic else 'original SAV labels; original CSV cross-check'}\n"
        'Target: Q30 ai_heard substantive codes 1/2/3; Q30 label-defined 8/9 removed.\n'
        'Predictors: all 483 raw columns audited. Skip identifiers, survey weight, target, survey field,\n'
        'date strings, single-level variables, >25-category variables and <100 joint complete cases.\n'
        'Predictor nonresponse codes identified from THAT variable\'s original SAV value labels.\n'
        'No global 8/9 recode: Q32 8 Not sure and internetfreq 6 non-user remain real answers.\n'
        'Formula: sqrt((Pearson chi-square / N)/(min(number_x_levels-1,number_y_levels-1)));\n'
        'chi2_contingency(..., correction=False). Unweighted, pairwise complete case, no bias correction.\n'
        'Age is the separately derived 4-category age_band, not among the 483 raw columns.\n'
        'Age-band comparison includes 111 recovered followups and excludes 185 missing bands.\n'
        'Reference calculations: internetfreq N=25625, V=0.259616661; age_band N=27340, V=0.085108134.\n'
        'Earlier checklist suggested age V~0.22, which is NOT replicated with same estimator and groupings.\n'
        'Some original raw columns are asked only in certain countries; ranking across varying Ns is\n'
        'not a controlled comparison. Very sparse tables are flagged. Do not infer significance,\n'
        'causality or full-survey design variance from this unadjusted multiple-comparison screen.\n'
        'French d_income2_france code 98 (5 records) remains an unlabelled data-quality exception;\n'
        'its exploratory association is not substantively interpreted pending codebook clarification.\n'
        'Original files and Task 29 provisional cleaned data were NOT overwritten. P2 educ3 remains pending.\n'
    )
    (OUT/f'{prefix}validation_note.txt').write_text(note,encoding='utf-8')
    d={ 'task':32,'decision':'Use unweighted Cramers V consistently for all categorical scan predictors and the derived four-group age band',
        'alternative':'Mix Pearson correlation of continuous age with Cramers V for internet or accept 0.22 age claim unverified',
        'why_rejected':'Comparing distinct estimators/categories produces noncomparable figures; the same estimator gives age V=0.085108134',
        'effect':'Scan each of 483 raw fields; rank eligible columns; compare age V~0.085 to internet V~0.260',
        'limitations':'exploratory, unadjusted, unweighted, repeated testing, varying complete cases/structural missingness',
        'status':'CSV diagnostic only' if diagnostic else 'original SAV and CSV verified'}
    pd.DataFrame([d]).to_csv(OUT/f'{prefix}decision_log_entry.csv',index=False)
    pd.DataFrame([{'task':32,'ai_contribution':'Generated full-scan validation code, codebook-driven missingness, mathematical self-tests and diagnostic comparison',
                   'human_validation':'CSV-only diagnostic' if diagnostic else 'Student original SAV and original CSV run; inspect output',
                   'ai_error_review':'Earlier projected age association V~0.22 was unsupported under same-metric test',
                   'source':'Original Pew SAV metadata and original CSV'}]).to_csv(
                       OUT/f'{prefix}ai_use_log_entry.csv',index=False)
    pd.DataFrame([
        {'task':32,'earlier_ai_claim':'Age association approximately 0.22',
         'discrepancy':'Four-level age_band vs substantive Q30 using unweighted Cramers V yields 0.085108134; internetfreq yields 0.259616661',
         'detected_by':'Recomputing both with identical formula and original data',
         'correction':'Report 0.085 age vs 0.260 internet with separate Ns and unadjusted exploratory caveats',
         'status':'CORRECTED'},
        {'task':32,'earlier_ai_claim':'Initial validator development code would run without syntax or ID dtype issues',
         'discrepancy':'First draft had a misquoted string; SAV float ID vs CSV integer ID could also incorrectly fail pandas .equals',
         'detected_by':'py_compile/self-test and portability review before release',
         'correction':'Fixed string quoting and compare integer respondent ID values rather than dtype-sensitive Series.equals',
         'status':'CORRECTED_BEFORE_HANDOFF'}
    ]).to_csv(OUT/f'{prefix}ai_discrepancy_entry.csv',index=False)
    print(f'Saved {prefix}full_483_variable_scan.csv, {prefix}associations_ranked.csv,')
    print(f'      {prefix}key_comparisons.csv, logs and validation note (AGGREGATES ONLY)')


def self_test():
    x=pd.Series([1,1,2,2,6,6,8,9],dtype=float)
    y=pd.Series([1,1,2,2,3,3,8,9],dtype=float)
    fake=SimpleNamespace(variable_value_labels={
        'ai_heard':{1:'A lot',2:'A little',3:'Nothing at all',8:"Don't know",9:'Refused'},
        'internetfreq':{1:'Almost constantly',2:'Several times a day',6:'does not use the internet',8:"Don't know",9:'Refused'},
        'aireg_us':{1:'A lot of trust',8:'Not sure',9:'Refused'}})
    raw=pd.DataFrame({'id':range(1,9),'ai_heard':y,'internetfreq':x})
    target=q30_target(raw,fake)
    good=missing_masked(raw.internetfreq,fake,'internetfreq')
    assert good.eq(6).sum()==2 and good.isna().sum()==2
    assert target.isna().sum()==2
    # All 6 valid cases align exactly across the 3 substantive categories.
    v=cramers_v(good,target,min_n=4)
    assert v['status']=='tested' and v['n_joint_complete']==6 and np.isclose(v['cramers_v'],1)
    print('PASS: non-users preserved, Q30 and internet DK/refused excluded; perfect synthetic V=1')
    r=missing_masked(pd.Series([1,8,9],dtype=float),fake,'aireg_us')
    assert r.eq(8).sum()==1 and r.isna().sum()==1
    print('PASS: Q32 code 8 Not sure retained, only labelled refusal excluded')
    xx=pd.Series([1,2,1,2],dtype=float)
    yy=pd.Series([1,1,2,2],dtype=float)
    v=cramers_v(xx,yy,min_n=4)
    assert np.isclose(v['cramers_v'],0)
    print('PASS: unweighted Pearson chi-square Cramers V is 0 for independent 2x2')
    yy=pd.Series([1,1,2,2],dtype=float); xx=pd.Series([1,1,2,2],dtype=float)
    v=cramers_v(xx,yy,min_n=4)
    assert np.isclose(v['cramers_v'],1)
    print('PASS: uncorrected Cramers V is 1 for perfect 2x2')
    assert cramers_v(pd.Series([1,2,3,4]), yy, min_n=4,max_levels=3)['status'].startswith('skip_high')
    assert cramers_v(pd.Series([1,2,3,4]), yy,min_n=5)['status'].startswith('skip_fewer')
    print('PASS: skip rules correctly identify high cardinality and sparse samples')
    try:
        numeric(pd.Series(['1','unexpected']),label='untrusted')
    except ValueError: pass
    else: raise AssertionError('Unexpected raw text silently discarded')
    try:
        cramers_v(xx.reset_index(drop=True), yy.set_axis([3,2,1,0]),min_n=4)
    except AssertionError: pass
    else: raise AssertionError('Misaligned indexes went undetected')
    print('PASS: nonnumeric inputs and misaligned respondent indexes rejected')
    print('TASK 32 SELF-TEST PASSED')


def run(diagnostic:bool, labels_json:Path|None):
    print('=== TASK 32: ALL 483 SOURCE COLUMNS — CATEGORICAL ASSOCIATION SCREEN ===')
    print('Mode: CSV/dictionary diagnostic only' if diagnostic else 'Mode: ORIGINAL SAV + CSV')
    sav,csv,meta=load_sources(diagnostic,labels_json)
    assert sav.shape==csv.shape==(N,RAW_VARIABLES)
    assert sav.columns.equals(csv.columns), 'SAV/CSV original column names differ'
    assert len(sav.id)==len(csv.id) and sav.id.is_unique and csv.id.is_unique
    assert np.array_equal(numeric(sav.id,label='SAV id').to_numpy(dtype=np.int64),
                          numeric(csv.id,label='CSV id').to_numpy(dtype=np.int64)), (
        'SAV and CSV respondent ID order differs; check alignment before comparing')
    assert q30_target(sav,meta).value_counts(dropna=False).loc[[1,2,3]].astype(int).to_dict() == {1:9407,2:12878,3:5240}
    print('PASS: source rows, IDs and original columns match (28,333 x 483)')
    sav_untouched=sav[['id','ai_heard','internetfreq','weight']].copy(deep=True)
    csv_untouched=csv[['id','ai_heard','internetfreq','weight']].copy(deep=True)
    sav_clean=do_clean(sav,meta,[])
    csv_clean=do_clean(csv,meta,[])
    compare_critical_by_id(sav,csv,sav_clean,csv_clean)
    if not diagnostic:
        optional_export_integrity(sav_clean)
    scan_sav=scan_483(sav,meta)
    scan_csv=scan_483(csv,meta)
    compare_scan(scan_sav,scan_csv)
    age_sav=unified_age_record(sav,sav_clean,meta)
    age_csv=unified_age_record(csv,csv_clean,meta)
    assert age_sav['n_joint_complete']==age_csv['n_joint_complete']
    assert np.isclose(age_sav['cramers_v'],age_csv['cramers_v'],rtol=0,atol=1e-12)
    verify_reference(scan_sav,age_sav)
    verify_reference(scan_csv,age_csv)
    independent_existing_function(sav,meta,scan_sav)
    pd.testing.assert_frame_equal(sav[['id','ai_heard','internetfreq','weight']],sav_untouched)
    pd.testing.assert_frame_equal(csv[['id','ai_heard','internetfreq','weight']],csv_untouched)
    assert len(sav)==len(csv)==len(sav_clean)==len(csv_clean)==N
    print('PASS: original inputs unchanged; all 28,333 people retained; no files overwritten')
    eligible=scan_sav.loc[scan_sav.status.eq('tested')]
    skipped=scan_sav.loc[scan_sav.status.ne('tested')]
    print(f'Raw variables examined: {len(scan_sav)}; tested: {len(eligible)}; excluded with reason: {len(skipped)}')
    print('Exclusion reasons:',skipped.status.value_counts().to_dict())
    print('TOP ELIGIBLE EXPLORATORY ASSOCIATIONS (N>=1,000; varying sample sizes):')
    top=(eligible.loc[eligible.n_joint_complete.ge(1000)]
         .sort_values('cramers_v',ascending=False)
         [['variable','n_joint_complete','n_x_levels_in_table','cramers_v']].head(10))
    print(top.to_string(index=False,float_format=lambda z:f'{z:.4f}'))
    print(f"Focus: internetfreq N={int(scan_sav.loc[scan_sav.variable.eq('internetfreq'),'n_joint_complete'].iloc[0]):,}, "
          f"V={scan_sav.loc[scan_sav.variable.eq('internetfreq'),'cramers_v'].iloc[0]:.6f}; "
          f"final age_band N={age_sav['n_joint_complete']:,}, V={age_sav['cramers_v']:.6f}")
    write_outputs(scan_sav,age_sav,diagnostic=diagnostic)
    if diagnostic:
        print('TASK 32 CSV DIAGNOSTIC PASSED (not yet verified against original SAV on your PC)')
    else:
        print('TASK 32 PASSED: 483/483 original columns audited; SAV/CSV association scan agrees;')
        print('                same-metric internet V~0.260 and age-band V~0.085; no data changed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--csv-diagnostic',action='store_true')
    parser.add_argument('--labels-json',type=Path)
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.csv_diagnostic,args.labels_json)
