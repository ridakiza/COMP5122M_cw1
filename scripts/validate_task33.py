"""COMP5122M Task 33 — validate what `phone_sample` really means.

From repo root:
    python scripts/validate_task33.py --self-test
    python scripts/validate_task33.py

Optional CSV-only diagnostic (NOT original-SAV validation):
    python scripts/validate_task33.py --csv-diagnostic --dictionary path/to/dictionary.xlsx

The official data dictionary and SAV labels define phone_sample as
'Telephone sample source': 1 Landline, 2 Cell. They do NOT say it is an
interview-mode flag. Blank country blocks cannot establish their interview mode.
No data file is modified. Outputs contain country aggregates only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
OUT = ROOT / 'data/clean'
FIELDS = ['id', 'country', 'survey', 'phone_sample']
EXPECTED_ROWS = 28_333
EXPECTED_COUNTRIES = 24
# Independent source-data reference counts; never used to translate codes.
EXPECTED_COUNTS = {'Landline': 3058, 'Cell': 10073, 'Not recorded': 15202}
EXPECTED_COVERAGE = {'countries_with_phone_source': 13,
                     'countries_without_phone_source': 11,
                     'countries_mixed_landline_cell': 10,
                     'countries_cell_only': 3}


def normalize_text(value: object) -> str:
    return ' '.join(str(value).casefold().replace('’', "'").split())


def lookup_labels(meta: object, variable: str) -> dict[int, str]:
    labels_by_var = getattr(meta, 'variable_value_labels', None) or {}
    matches = [val for key, val in labels_by_var.items()
               if str(key).strip().casefold() == variable.casefold()]
    assert len(matches) == 1, f'Missing/ambiguous SAV code labels for {variable}'
    values = {}
    for key, label in matches[0].items():
        numeric_code = float(key)
        assert numeric_code.is_integer(), f'Nonintegral code in {variable} metadata'
        k = int(numeric_code)
        assert k not in values, f'Duplicate metadata code in {variable}'
        values[k] = str(label).strip()
    return values


def verify_phone_metadata(meta: object) -> dict[int, str]:
    labels = lookup_labels(meta, 'phone_sample')
    assert set(labels) == {1, 2}, f'Unexpected phone_sample SAV codes: {labels}'
    assert normalize_text(labels[1]) == 'landline', f'Code 1 is not Landline: {labels[1]}'
    assert normalize_text(labels[2]) in ('cell', 'cell phone', 'mobile', 'mobile phone'), (
        f'Code 2 is not Cell: {labels[2]}')
    descriptions = getattr(meta, 'column_names_to_labels', {}) or {}
    matches = [label for key, label in descriptions.items()
               if str(key).strip().casefold() == 'phone_sample']
    if matches and matches[0] is not None and str(matches[0]).strip():
        definition = normalize_text(matches[0])
        assert 'telephone' in definition and 'sample' in definition and 'source' in definition, (
            f'Original SAV variable description is unexpected: {matches[0]!r}')
    return labels


def normalize_frame(source: pd.DataFrame, name: str,
                    required_n: int | None = EXPECTED_ROWS) -> pd.DataFrame:
    assert set(FIELDS).issubset(source.columns), f'{name}: missing required columns'
    df = source[FIELDS].copy(deep=True)
    if required_n is not None:
        assert len(df) == required_n, f'{name}: expected {required_n} records, got {len(df)}'
    for col in ('id', 'country', 'survey'):
        value = pd.to_numeric(df[col], errors='coerce')
        assert value.notna().all() and np.isfinite(value.to_numpy(dtype=float)).all(), (
            f'{name}: missing/non-numeric {col}')
        assert value.mod(1).eq(0).all(), f'{name}: nonintegral {col}'
        df[col] = value.astype('int64')
    x = df.phone_sample
    if not pd.api.types.is_numeric_dtype(x):
        x = x.astype('string').str.strip().replace('', pd.NA)
    converted = pd.to_numeric(x, errors='coerce')
    assert converted.notna().equals(x.notna()), f'{name}: unexpected nonnumeric phone_sample'
    assert set(converted.dropna().astype(int).unique()).issubset({1, 2}), (
        f'{name}: unexpected phone_sample code: {set(converted.dropna().unique())}')
    df['phone_sample'] = converted.astype('Float64')
    assert df.id.is_unique, f'{name}: repeated respondent IDs'
    assert (df.survey == 1202501).all() or required_n is None, (
        f'{name}: source is not exclusively Spring 2025 survey')
    return df


def compare_by_id(first: pd.DataFrame, second: pd.DataFrame) -> None:
    a = normalize_frame(first, 'SAV').set_index('id').sort_index()
    b = normalize_frame(second, 'CSV').set_index('id').sort_index()
    assert a.index.equals(b.index), 'SAV/CSV respondent IDs differ'
    for field in ('country', 'survey', 'phone_sample'):
        x = a[field].to_numpy(dtype=float, na_value=np.nan)
        y = b[field].to_numpy(dtype=float, na_value=np.nan)
        assert np.isclose(x, y, rtol=0, atol=0, equal_nan=True).all(), (
            f'SAV/CSV discrepancy in {field}')
    print(f'PASS: original SAV/CSV agree on all {len(a):,} IDs (country, survey, phone_sample)')


def get_country_names(meta: object, observed: set[int],
                      require_24: bool = True) -> dict[int, str]:
    labels = lookup_labels(meta, 'country')
    assert set(labels) == observed, 'SAV country labels do not match observed country codes'
    if require_24:
        assert len(labels) == EXPECTED_COUNTRIES
    assert len(set(labels.values())) == len(labels), 'Duplicate country names'
    return labels


def audit(source: pd.DataFrame, meta: object, *, name: str,
          strict: bool = True) -> tuple[pd.DataFrame, dict]:
    labels = verify_phone_metadata(meta)
    df = normalize_frame(source, name, EXPECTED_ROWS if strict else None)
    country_names = get_country_names(meta, set(df.country.unique()), require_24=strict)
    assert df.survey.nunique() == 1, f'{name}: survey field is not constant'
    assert lookup_labels(meta, 'survey')[int(df.survey.iloc[0])].strip(), (
        f'{name}: survey metadata has no label')
    rows = []
    for code, sub in df.groupby('country', sort=True):
        count_landline = int(sub.phone_sample.eq(1).fillna(False).sum())
        count_cell = int(sub.phone_sample.eq(2).fillna(False).sum())
        count_blank = int(sub.phone_sample.isna().sum())
        n = len(sub)
        assert n == count_landline + count_cell + count_blank
        assert count_blank in (0, n), (
            f'{name}: country {country_names[code]} has partial phone_sample coverage; investigate')
        coverage = ('not_recorded' if count_blank == n else
                    'mixed_landline_cell' if count_landline and count_cell else
                    'cell_only' if count_cell else 'landline_only')
        rows.append({'country_code': int(code), 'country':country_names[code],
                     'respondents':n, 'landline_source_n':count_landline,
                     'cell_source_n':count_cell, 'phone_source_unrecorded_n':count_blank,
                     'landline_source_pct_of_country':100*count_landline/n,
                     'cell_source_pct_of_country':100*count_cell/n,
                     'coverage':coverage,
                     'meaning':'Sampling frame/source; NOT interview mode'})
    countries = pd.DataFrame(rows)
    observed_counts = {'Landline':int(countries.landline_source_n.sum()),
                       'Cell':int(countries.cell_source_n.sum()),
                       'Not recorded':int(countries.phone_source_unrecorded_n.sum())}
    coverage = countries.coverage.value_counts().to_dict()
    observed_coverage = {'countries_with_phone_source':int(countries.phone_source_unrecorded_n.eq(0).sum()),
                         'countries_without_phone_source':int(countries.coverage.eq('not_recorded').sum()),
                         'countries_mixed_landline_cell':int(coverage.get('mixed_landline_cell',0)),
                         'countries_cell_only':int(coverage.get('cell_only',0))}
    assert sum(observed_counts.values()) == len(df)
    if strict:
        assert len(df) == EXPECTED_ROWS and len(countries) == EXPECTED_COUNTRIES
        assert observed_counts == EXPECTED_COUNTS, f'{name}: unexpected distribution {observed_counts}'
        assert observed_coverage == EXPECTED_COVERAGE, (
            f'{name}: unexpected country-level phone sampling coverage {observed_coverage}')
    result = {'respondents':len(df),'survey_code':int(df.survey.iloc[0]),
              'survey_value_label':lookup_labels(meta,'survey')[int(df.survey.iloc[0])],
              'variable':'phone_sample',
              'variable_description':'Telephone sample source',
              'landline_code':1,'landline_label':labels[1],
              'cell_code':2,'cell_label':labels[2],
              'landline_n':observed_counts['Landline'],
              'cell_n':observed_counts['Cell'],
              'not_recorded_n':observed_counts['Not recorded'],
              **observed_coverage,
              'survey_interview_mode_available_from_phone_sample':'NO',
              'unknown_countries_interview_modes':'Not inferable from this column',
              'data_changed':'NO'}
    return countries,result


def assert_country_matches(a: pd.DataFrame, b: pd.DataFrame) -> None:
    a = a.set_index('country_code').sort_index()
    b = b.set_index('country_code').sort_index()
    assert a.index.equals(b.index), 'Country code routing differs'
    cols = ['country','respondents','landline_source_n','cell_source_n',
            'phone_source_unrecorded_n','coverage']
    for col in cols:
        assert a[col].tolist() == b[col].tolist(), f'Country field differs: {col}'


def load_sav_csv() -> tuple[pd.DataFrame,pd.DataFrame,object]:
    if not CSV.exists():
        raise FileNotFoundError(f'Missing original CSV: {CSV}')
    try:
        import pyreadstat
    except ImportError as e:
        raise RuntimeError('Install pyreadstat in your venv to verify original SAV') from e
    if not SAV.exists():
        raise FileNotFoundError(f'Missing original SAV: {SAV}')
    sav, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False,user_missing=True)
    sav.columns = [str(x).strip().lower() for x in sav.columns]
    csv = pd.read_csv(CSV,low_memory=False,encoding='utf-8-sig')
    csv.columns = [str(x).strip().lower() for x in csv.columns]
    assert sav.shape == csv.shape == (EXPECTED_ROWS,483), 'Unexpected source dimensions'
    return sav, csv, meta


def load_dictionary_metadata(path:Path) -> object:
    """Dictionary proxy for local independent CSV check ONLY; not SAV metadata."""
    if not path.is_file():
        raise FileNotFoundError(path)
    vals = pd.read_excel(path, sheet_name='Values',header=None).iloc[2:,:3]
    vals.columns = ['variable','code','label']
    vals['variable'] = vals['variable'].ffill()
    selected = vals[vals.variable.isin(['phone_sample','country','survey'])]
    labels = {}
    for v, subset in selected.groupby('variable'):
        labels[v] = {int(float(r.code)):str(r.label) for r in subset.itertuples(index=False)}
    var = pd.read_excel(path,sheet_name='Variables',header=None).iloc[2:,:3]
    row = var[var.iloc[:,0].eq('phone_sample')]
    assert len(row)==1
    label = str(row.iloc[0,2])
    return SimpleNamespace(variable_value_labels=labels,
                           column_names_to_labels={'phone_sample':label})


def self_test():
    meta = SimpleNamespace(variable_value_labels={
        'phone_sample':{1:'Landline',2:'Cell'},
        'country':{2:'Canada',3:'France'},
        'survey':{1202501:'Spring, 2025'}},
        column_names_to_labels={'phone_sample':'Telephone sample source'})
    tiny = pd.DataFrame({'id':[11,12,13,14], 'country':[2,2,2,3],
                         'survey':[1202501]*4, 'phone_sample':[1,2,1,np.nan]})
    snapshot = tiny.copy(deep=True)
    country, totals = audit(tiny,meta,name='SYNTHETIC',strict=False)
    assert totals['landline_n']==2 and totals['cell_n']==1 and totals['not_recorded_n']==1
    assert country.loc[country.country.eq('France'),'coverage'].iloc[0]=='not_recorded'
    pd.testing.assert_frame_equal(tiny,snapshot)
    print('PASS: 1 Landline, 2 Cell distinguished as sample source; blanks preserved')
    compare = normalize_frame(tiny,'SYNTHETIC',required_n=None)
    assert len(compare)==4
    # Challenge metadata labels (reject modes replacing sampling-source meanings).
    wrong_meta = SimpleNamespace(variable_value_labels={**meta.variable_value_labels,
        'phone_sample':{1:'Telephone interview',2:'Online interview'}},
        column_names_to_labels=meta.column_names_to_labels)
    try: audit(tiny,wrong_meta,name='SYNTHETIC',strict=False)
    except AssertionError: print('PASS: incorrect interview-mode interpretation rejected by original value labels')
    else: raise AssertionError('Failed to reject mislabelled phone_sample')
    # Reject a third (undocumented) response code.
    bad = tiny.copy(); bad.loc[0,'phone_sample']=3
    try: audit(bad,meta,name='SYNTHETIC',strict=False)
    except AssertionError: print('PASS: unknown sample-source code rejected')
    else: raise AssertionError('Failed to reject undocumented code')
    # Reject partial country coverage (normally structural by country).
    partial = tiny.copy(); partial.loc[0,'phone_sample']=np.nan
    try: audit(partial,meta,name='SYNTHETIC',strict=False)
    except AssertionError: print('PASS: unexpected partial-country missingness detected')
    else: raise AssertionError('Failed to reject partial-country source coverage')
    # Reject respondent-level source mismatch, not just aggregate counts.
    mismatch = tiny.copy(); mismatch.loc[0,'phone_sample']=2
    a = normalize_frame(tiny,'A',required_n=None).set_index('id')
    b = normalize_frame(mismatch,'B',required_n=None).set_index('id')
    assert not np.isclose(a.phone_sample.to_numpy(dtype=float,na_value=np.nan),
                          b.phone_sample.to_numpy(dtype=float,na_value=np.nan),
                          rtol=0,atol=0,equal_nan=True).all()
    print('PASS: respondent-level source-code mismatch detected')
    print('TASK 33 SELF-TEST PASSED')


def save_outputs(country:pd.DataFrame,summary:dict, *,diagnostic:bool) -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    prefix='task33_csv_diagnostic_' if diagnostic else 'task33_'
    country.to_csv(OUT/f'{prefix}phone_sample_country_audit.csv',index=False)
    pd.DataFrame([summary]).to_csv(OUT/f'{prefix}phone_sample_summary.csv',index=False)
    decision = {'task':33,
        'decision':'Interpret phone_sample as the telephone sample source: 1 Landline, 2 Cell',
        'evidence':'Original SAV value labels, dictionary field description Telephone sample source, 24-country coverage audit',
        'alternative':'Use it as a general survey interview-mode indicator (e.g. phone vs online/in person)',
        'why_rejected':'Sampling source/frame differs from interview mode; the variable supplies only Landline/Cell codes and is blank in 11 countries',
        'effect':'13 countries have recorded telephone source, 11 have no phone_sample value; no interview mode is inferred',
        'data_effect':'No recoding, row removal, or overwrite',
        'status':'CSV diagnostic only' if diagnostic else 'SAV and CSV verified'}
    pd.DataFrame([decision]).to_csv(OUT/f'{prefix}decision_log_entry.csv',index=False)
    ai = {'task':33,
          'ai_contribution':'Generated label-driven interpretation audit, independent counts, country-routing checks, respondent-level source comparison and synthetic tests',
          'reference':'Pew original SAV metadata and provided Excel data dictionary',
          'validation':'CSV/dictionary only' if diagnostic else 'Student to confirm original SAV and original CSV run',
          'ai_errors':'None found in final Task 33 validation'}
    pd.DataFrame([ai]).to_csv(OUT/f'{prefix}ai_use_log_entry.csv',index=False)
    note = (
        'TASK 33 — PHONE_SAMPLE INTERPRETATION AUDIT\n'
        f"Source: {'CSV + dictionary proxy only (not authoritative SAV)' if diagnostic else 'original SAV labels and CSV respondent-level agreement'}\n"
        'Codebook description: phone_sample = Telephone sample source.\n'
        'Original value labels: 1 Landline; 2 Cell. There are no labels for phone/online/in-person interview mode.\n'
        'Original survey variable is a separate constant Spring 2025 survey identifier, not interview mode.\n'
        f"Recorded: Landline={summary['landline_n']}, Cell={summary['cell_n']}, "
        f"Not recorded={summary['not_recorded_n']} out of {summary['respondents']} respondents.\n"
        f"Country coverage: {summary['countries_with_phone_source']} with phone source ("
        f"{summary['countries_mixed_landline_cell']} both, {summary['countries_cell_only']} cell only); "
        f"{summary['countries_without_phone_source']} countries with no phone_sample.\n"
        'The missing phone_sample values are entire country blocks; do not interpret them as a third mode.\n'
        'A landline/cell sampling-source comparison can be explored within eligible countries, '
        'but cannot be labelled a phone-versus-online survey-mode effect or causally interpreted.\n'
        'The dataset alone does not establish interview modes for countries with a blank source field.\n'
        'No original/cleaned respondent-level files were modified; outputs contain AGGREGATES only.\n'
    )
    (OUT/f'{prefix}validation_note.txt').write_text(note,encoding='utf-8')
    print('Saved: '+f'{prefix}phone_sample_country_audit.csv, '+f'{prefix}phone_sample_summary.csv,')
    print('       decision/AI-use log entries and validation note (AGGREGATES ONLY)')


def run(*,diagnostic:bool=False,dictionary:Path|None=None):
    print('=== TASK 33: PHONE_SAMPLE — SAMPLING SOURCE, NOT INTERVIEW MODE ===')
    if diagnostic:
        assert dictionary is not None, '--csv-diagnostic requires --dictionary PATH'
        source = pd.read_csv(CSV,low_memory=False,encoding='utf-8-sig')
        source.columns = [str(x).strip().lower() for x in source.columns]
        meta = load_dictionary_metadata(dictionary)
        assert source.shape==(EXPECTED_ROWS,483)
        left,first = audit(source,meta,name='CSV',strict=True)
        print('NOTE: CSV + Excel dictionary diagnostic, NOT original SAV validation')
    else:
        sav,csv,meta = load_sav_csv()
        sav_before=sav[FIELDS].copy(deep=True)
        csv_before=csv[FIELDS].copy(deep=True)
        compare_by_id(sav,csv)
        left,first=audit(sav,meta,name='SAV',strict=True)
        right,other=audit(csv,meta,name='CSV',strict=True)
        assert_country_matches(left,right)
        assert first==other, 'SAV/CSV summary mismatch'
        pd.testing.assert_frame_equal(sav[FIELDS],sav_before)
        pd.testing.assert_frame_equal(csv[FIELDS],csv_before)
        print('PASS: 24/24 country sample-source audits agree between SAV/CSV')
        print('PASS: source SAV/CSV unchanged; 28,333 respondents retained')
    print('SAV/dictionary variable: phone_sample — Telephone sample source')
    print('Value labels: {1: Landline, 2: Cell}')
    print('Observed respondents: Landline=3,058; Cell=10,073; Not recorded=15,202')
    print('Countries: 13 record phone source (10 both frames, 3 cell-only); 11 entirely blank')
    print('The original survey field is constant Spring 2025; it does not encode interview mode')
    print('Country sample-source audit:')
    print(left[['country','respondents','landline_source_n','cell_source_n',
                'phone_source_unrecorded_n','coverage']].to_string(index=False))
    print('CONCLUSION: phone_sample identifies landline vs cell sampling source, NOT survey interview mode.')
    print('CAUTION: no claim about interview modes can be made from the 15,202 blanks.')
    save_outputs(left,first,diagnostic=diagnostic)
    if diagnostic:
        print('TASK 33 CSV DIAGNOSTIC PASSED; original SAV verification still required.')
    else:
        print('TASK 33 PASSED: metadata meanings, 24-country routing and all 28,333 SAV/CSV respondents verified.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--csv-diagnostic',action='store_true')
    p.add_argument('--dictionary',type=Path,default=None)
    args=p.parse_args()
    if args.self_test:
        self_test()
    else:
        run(diagnostic=args.csv_diagnostic,dictionary=args.dictionary)
