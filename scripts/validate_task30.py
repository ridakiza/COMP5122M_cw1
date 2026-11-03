"""COMP5122M P1 — Task 30: prepare and audit P2 educ3 integration.

Run from repository root:
  python scripts/validate_task30.py --self-test
  python scripts/validate_task30.py --preflight

After P2 supplies *id,educ3* for every respondent:
  python scripts/validate_task30.py --p2 data/clean/education_from_p2.csv

After reviewing P2's country-by-country category rules, document them in a
private JSON schema and opt in to writing the final merged dataset:
  python scripts/validate_task30.py --p2 data/clean/education_from_p2.csv \
      --schema data/clean/p2_educ3_schema.json --export-final

NO P2 DATA IS PROVIDED HERE. The self-test uses invented, explicitly synthetic
values. Preflight reports PENDING, not PASSED, until real P2 data is available.
Never commit respondent-level CSVs, schemas or aggregate country results
containing small cells to a public repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from validate_task19 import load_reference  # previously independently verified official 192-cell reference
from validate_task29 import preflight_git_protection

CLEAN = ROOT / 'data' / 'clean'
P1_PATH = CLEAN / 'pew_clean_p1_provisional.csv'
P1_MANIFEST = CLEAN / 'task29_provisional_export_manifest.csv'
REF = ROOT / 'references' / 'topline_expected.csv'
FINAL = CLEAN / 'pew_clean_p1_p2_final.csv'
N = 28_333


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_ids(series: pd.Series, source: str) -> pd.Series:
    """Require unique positive integer IDs, not loosely matched strings/floats."""
    strings = series.astype('string').str.strip()
    if strings.isna().any() or strings.eq('').any():
        raise ValueError(f'{source}: missing or empty respondent ID')
    try:
        nums = pd.to_numeric(strings, errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{source}: nonnumeric respondent ID') from exc
    arr = nums.to_numpy(dtype='float64', na_value=np.nan)
    if not np.isfinite(arr).all() or not np.equal(arr, np.floor(arr)).all() or (arr <= 0).any():
        raise ValueError(f'{source}: respondent IDs must be positive integers')
    if (arr > 2**53).any():
        raise ValueError(f'{source}: respondent ID exceeds safe exact numeric range')
    ids = pd.Series(arr.astype('int64'), index=series.index, name='id')
    if ids.duplicated().any():
        raise ValueError(f'{source}: {int(ids.duplicated().sum())} duplicate IDs')
    return ids


def read_p1(path: Path = P1_PATH, *, require_manifest: bool = True) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Task 29 provisional export missing: {path}. Run Task 29 first.')
    if require_manifest:
        if not P1_MANIFEST.is_file():
            raise FileNotFoundError('Task 29 SHA-256 manifest missing; rerun Task 29 to generate it.')
        manifest = pd.read_csv(P1_MANIFEST)
        if len(manifest) != 1 or manifest.loc[0, 'export_filename'] != path.name:
            raise ValueError('Task 29 manifest does not identify the provisional P1 CSV')
        expected = str(manifest.loc[0, 'sha256']).strip().lower()
        observed = sha256(path)
        if observed != expected:
            raise ValueError('Task 29 provisional dataset SHA-256 has changed. Do not merge.')
        if int(manifest.loc[0, 'rows']) != N or int(manifest.loc[0, 'columns']) != 70:
            raise ValueError('Task 29 manifest row/column counts unexpected')
        print('PASS: Task 29 export checksum matches its recorded manifest')
    df = pd.read_csv(path, low_memory=False, encoding='utf-8-sig')
    if len(df) != N or len(df.columns) != 70 or 'educ3' in df:
        raise ValueError(f'Expected Task 29 P1 export: 28,333 × 70, no educ3; got {df.shape}')
    required = {'id','weight','country','country_name','ai_heard','ai_cncexc',
                'age_band','age_band_recovered','gender_clean'}
    if not required.issubset(df.columns):
        raise ValueError(f'P1 data lacks required fields: {sorted(required - set(df.columns))}')
    df['id'] = canonical_ids(df['id'], 'P1')
    if int(df.age_band.isna().sum()) != 185 or int(df.age_band_recovered.sum()) != 111:
        raise ValueError('P1 known age-band/missingness totals do not match Task 29')
    if int(df.gender_clean.eq(3).sum()) != 21:
        raise ValueError('P1 Australian Other responses changed')
    w = pd.to_numeric(df.weight, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(w).all() or not (w > 0).all():
        raise ValueError('P1 weights invalid')
    print('PASS: P1 dataset 28,333 × 70; age, gender and weight controls intact')
    return df


def read_p2(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'P2 education lookup NOT received: {path}')
    p2 = pd.read_csv(path, dtype='string', keep_default_na=False, encoding='utf-8-sig')
    p2.columns = p2.columns.str.strip()
    if p2.columns.duplicated().any() or not {'id','educ3'}.issubset(p2.columns):
        raise ValueError('P2 CSV must contain unique headers id and educ3')
    extras = [col for col in p2 if col not in ('id','educ3')]
    if extras:
        print('NOTE: P2 supplied extra columns; they are NOT merged:', extras)
    p2 = p2[['id','educ3']].copy()
    p2['id'] = canonical_ids(p2.id, 'P2')
    p2['educ3'] = p2.educ3.str.strip().replace('', pd.NA)
    if p2.educ3.isna().all():
        raise ValueError('All P2 education values are missing')
    if p2['educ3'].dropna().nunique() < 2:
        raise ValueError('P2 educ3 has fewer than two observed categories; investigate')
    return p2


def merge_checked(p1: pd.DataFrame, p2: pd.DataFrame) -> pd.DataFrame:
    if 'educ3' in p1:
        raise ValueError('P1 already contains educ3 — refusing to overwrite a previous mapping')
    for source, df in [('P1',p1),('P2',p2)]:
        if df.id.isna().any() or not df.id.is_unique:
            raise ValueError(f'{source}: missing/duplicate IDs')
    a, b = set(p1.id), set(p2.id)
    missing, extra = a-b, b-a
    if missing or extra:
        raise ValueError(f'P2 respondent coverage incorrect: missing={len(missing)}, extra={len(extra)}; '
                         'require an explicit row for EVERY P1 ID, even when educ3 is blank')
    result = p1.merge(p2, on='id', how='left', sort=False, validate='one_to_one', indicator=True)
    if not (result._merge == 'both').all():
        raise AssertionError('Unmatched IDs after merge')
    result = result.drop(columns='_merge')
    if result.shape != (len(p1), len(p1.columns)+1):
        raise AssertionError('Merged row/column counts changed unexpectedly')
    if result.columns.tolist() != p1.columns.tolist() + ['educ3']:
        raise AssertionError('P1 column order or content changed')
    # Same P1 values, same order: the only new field must be educ3.
    pd.testing.assert_frame_equal(result[p1.columns], p1, check_dtype=True)
    if result.educ3.notna().sum() != p2.educ3.notna().sum():
        raise AssertionError('P2 education nonmissing counts changed')
    print('PASS: strict one-to-one ID join, ALL P1 fields preserved; 28,333 rows, 71 columns')
    return result


def read_schema(path: Path | None) -> tuple[set[str] | None, dict | None]:
    """Never invent P2 categories; require P2-authored, reviewed labels for export."""
    if path is None:
        return None, None
    if not path.is_file():
        raise FileNotFoundError(f'P2 education schema not found: {path}')
    obj = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj, dict) or not isinstance(obj.get('value_labels'), dict):
        raise ValueError('Schema must contain object value_labels mapping P2 codes to descriptions')
    labels = obj['value_labels']
    if len(labels) < 2 or any(not str(k).strip() or not str(v).strip() for k,v in labels.items()):
        raise ValueError('Schema must document at least two nonempty code->label entries')
    if obj.get('country_rules_verified') is not True:
        raise ValueError('P2 must confirm country_rules_verified=true after reviewing harmonisation rules')
    if not str(obj.get('method_source', '')).strip():
        raise ValueError('Schema must record a method_source referring to P2 documentation')
    return {str(k) for k in labels}, obj


def audit_codes(merged: pd.DataFrame, allowed: set[str] | None) -> pd.DataFrame:
    values = merged.educ3.dropna().astype('string')
    if allowed is not None:
        unexpected = set(values.unique()) - allowed
        if unexpected:
            raise ValueError(f'Undocumented P2 educ3 categories: {sorted(unexpected)}')
    counts = merged.educ3.fillna('<MISSING>').value_counts(dropna=False)
    summary = pd.DataFrame({'educ3': counts.index.astype(str), 'respondents': counts.to_numpy(dtype=int)})
    print('P2 educ3 values (unweighted counts; coding meaning requires P2 documentation):')
    print(summary.to_string(index=False))
    if int(merged.educ3.isna().sum()):
        print('NOTE: missing educ3 retained as missing; not imputed')
    return summary


def round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def verify_toplines(merged: pd.DataFrame, *, reference: Path = REF) -> None:
    """Reproduce Task 19's original, ALL-response denominator from cleaned P1.

    Here Q30/Q31 8 and 9 have already become NA, so the reference combined
    DK/Refused category is precisely the missing category. The denominator
    nevertheless includes ALL respondents, including those with NA answers.
    """
    ref = load_reference(reference)
    groups = {(q, country): group for country, group in merged.groupby('country_name', sort=False)
              for q in ('ai_heard','ai_cncexc')}
    seen = 0
    for row in ref.itertuples(index=False):
        g = groups.get((row.question,row.country))
        if g is None:
            raise AssertionError(f'Missing published country/question {row.country}, {row.question}')
        w = pd.to_numeric(g.weight, errors='raise')
        response = pd.to_numeric(g[row.question], errors='coerce')
        if not response.dropna().isin([1,2,3]).all():
            raise AssertionError(f'Unknown cleaned {row.question} category')
        mask = response.isna() if row.code == 8 else response.eq(row.code)
        pct = 100.0 * float(w.loc[mask].sum()) / float(w.sum())
        got = round_half_up(pct)
        if got != row.expected:
            raise AssertionError(f'Published topline mismatch: {row.country} {row.question} '
                                 f'category {row.code}: got {got}, published {row.expected}')
        seen += 1
    if seen != 192:
        raise AssertionError(f'Expected 192 published topline cells, got {seen}')
    print('PASS: all 192/192 weighted Pew toplines retained (DK/refused included in denominator)')


def export_atomic(merged: pd.DataFrame, dest: Path, *, overwrite: bool = False) -> str:
    if dest.resolve() != FINAL.resolve() or dest.parent.resolve() != CLEAN.resolve():
        raise ValueError('Final export is permitted ONLY at data/clean/pew_clean_p1_p2_final.csv')
    fd, filename = tempfile.mkstemp(prefix='.task30_', suffix='.csv', dir=CLEAN)
    os.close(fd)
    tmp = Path(filename)
    try:
        merged.to_csv(tmp, index=False, encoding='utf-8')
        loaded = pd.read_csv(tmp, low_memory=False)
        if len(loaded) != len(merged) or loaded.columns.tolist() != merged.columns.tolist():
            raise AssertionError('Final CSV roundtrip changed columns or number of rows')
        # Cross-check the entire final file's fields after serialization.
        for col in merged.columns:
            if col in ('country_name','age_band','age_band_source','gender_label','educ3'):
                x = merged[col].astype('string').str.strip().fillna('<MISSING>')
                y = loaded[col].astype('string').str.strip().fillna('<MISSING>')
                if not x.equals(y):
                    raise AssertionError(f'Final CSV roundtrip changed column {col}')
            elif col == 'age_band_recovered':
                if not merged[col].astype(bool).equals(loaded[col].astype(bool)):
                    raise AssertionError(f'Final CSV roundtrip changed column {col}')
            else:
                x = pd.to_numeric(merged[col], errors='coerce').to_numpy(dtype='float64', na_value=np.nan)
                y = pd.to_numeric(loaded[col], errors='coerce').to_numpy(dtype='float64', na_value=np.nan)
                if not np.isclose(x,y,rtol=0,atol=1e-10,equal_nan=True).all():
                    raise AssertionError(f'Final CSV roundtrip changed column {col}')
        h = sha256(tmp)
        if dest.exists() and not overwrite:
            if sha256(dest) != h:
                raise FileExistsError('Different final export already exists — refusing to overwrite. '
                                      'Use --overwrite only after reviewing differences.')
            print('PASS: final export already exists and matches; no overwrite needed')
            return h
        os.replace(tmp, dest)
        print('PASS: merged CSV written locally to private data/clean/ (not uploaded or committed)')
        return h
    finally:
        tmp.unlink(missing_ok=True)


def log_evidence(merged: pd.DataFrame, p2_path: Path, schema_path: Path | None,
                 exported: bool, final_sha: str | None) -> None:
    CLEAN.mkdir(exist_ok=True, parents=True)
    summary = audit_codes(merged, None)
    summary.to_csv(CLEAN/'task30_educ3_distribution.csv', index=False)
    country_audit = (merged.assign(educ3_missing=merged.educ3.isna())
                     .groupby('country_name', dropna=False, sort=True)
                     .agg(respondents=('id','size'),
                          educ3_missing=('educ3_missing','sum'),
                          educ3_categories=('educ3','nunique')).reset_index())
    country_audit['educ3_valid'] = country_audit['respondents'] - country_audit['educ3_missing']
    if len(country_audit) != 24 or int(country_audit.respondents.sum()) != N:
        raise AssertionError('Unexpected country coverage during educ3 audit')
    if (country_audit.educ3_valid == 0).any():
        print('CAUTION: some countries have no mapped educ3; ask P2 to justify before finalising')
    country_audit.to_csv(CLEAN/'task30_educ3_country_audit.csv', index=False)
    rows = [{'task':30, 'status':'MERGE_VERIFIED' if exported else 'P2_MAPPING_CHECKED_PENDING_FINAL_REVIEW',
             'source_p1_file':P1_PATH.name, 'source_p2_file':p2_path.name,
             'p1_sha256':sha256(P1_PATH),'p2_sha256':sha256(p2_path),
             'schema_file':schema_path.name if schema_path is not None else '<PENDING>',
             'rows':len(merged), 'columns':len(merged.columns),
             'nonmissing_educ3':int(merged.educ3.notna().sum()),
             'missing_educ3':int(merged.educ3.isna().sum()),
             'validated_topline_cells':192,'final_export_created':exported,
             'final_sha256':final_sha or '', 'private_team_upload_confirmed':False}]
    pd.DataFrame(rows).to_csv(CLEAN/'task30_integration_manifest.csv', index=False)
    pd.DataFrame([{'task':30,'decision':'Merge only P2 documented educ3 by strict id, never infer education from raw country codes',
                   'alternative':'Impute/collapse d_educ country codes in P1 or merge positionally',
                   'reason':'Only P2 has authorised harmonisation and coding decisions',
                   'effect':'28,333 IDs unchanged; additional educ3 field; missing educ3 preserved',
                   'status':'Verified merged file' if exported else 'Mapping/policy review pending'}]).to_csv(
                       CLEAN/'task30_decision_log_entry.csv',index=False)
    pd.DataFrame([{'task':30,'ai_contribution':'Generated strict one-to-one integration validator and tests; no P2 education codes invented',
                   'verification':'Task29 SHA manifest, ID coverage, P1 field invariance, 192 Pew toplines, read-back',
                   'human_step':'P2 supplies id,educ3 and country harmonisation rules; reviewer checks schema; private team sharing',
                   'status':'VERIFIED' if exported else 'PENDING'}]).to_csv(CLEAN/'task30_ai_use_log_entry.csv',index=False)
    print('Saved: task30_educ3_distribution.csv, task30_educ3_country_audit.csv,')
    print('       task30_integration_manifest.csv,')
    print('       task30_decision_log_entry.csv, task30_ai_use_log_entry.csv')


def run(*, preflight: bool, p2_path: Path | None, schema_path: Path | None,
        export_final: bool, overwrite: bool) -> None:
    print('=== TASK 30: P2 EDUCATION LOOKUP / STRICT ID MERGE ===')
    preflight_git_protection()
    p1 = read_p1()
    if preflight:
        if p2_path is None or not p2_path.is_file():
            print('TASK 30 PENDING: P2 has not supplied an id,educ3 file; no mapping invented, no file written.')
            return
        print('NOTE: P2 file exists. Run --p2 FILE to validate its contents.')
        return
    if p2_path is None:
        raise SystemExit('Task 30 needs P2 file: use --preflight for readiness or --p2 PATH after receiving it.')
    p2 = read_p2(p2_path)
    merged = merge_checked(p1, p2)
    allowed, schema = read_schema(schema_path)
    audit_codes(merged, allowed)
    verify_toplines(merged)
    if export_final and schema is None:
        raise ValueError('Refusing final export: supply --schema with P2-authored value labels, '
                         'method source and confirmed country rules.')
    final_sha = export_atomic(merged, FINAL, overwrite=overwrite) if export_final else None
    log_evidence(merged, p2_path, schema_path, export_final, final_sha)
    if export_final:
        print('Final file:', FINAL.relative_to(ROOT))
        print('SHA-256:', final_sha)
        print('TASK 30 MERGE VERIFIED AND EXPORTED: 28,333 rows × 71 fields; education labels documented.')
    else:
        print('TASK 30 MAPPING VALIDATED STRUCTURALLY: no export; '
              'semantic P2 approval and explicit --export-final still required.')


def assert_raises(exception, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exception:
        return
    raise AssertionError(f'Expected {exception.__name__} was not raised')


def self_test() -> None:
    """Synthetic (invented!) IDs/categories only; nothing saved under data/clean."""
    p1 = pd.DataFrame({'id':[101,102,103,104], 'country':[1,1,2,2],
                       'weight':[1.,2.,0.8,1.2], 'ai_heard':[1.,np.nan,3.,2.],
                       'age_band':['18–29','65+','30–49','65+'],
                       'age_band_recovered':[False,False,True,False],
                       'gender_clean':[1.,2.,3.,np.nan]})
    p2 = pd.DataFrame({'id':[103,101,104,102],
                       'educ3':pd.Series(['3','1',pd.NA,'2'],dtype='string')})
    merged = merge_checked(p1,p2)
    assert merged.id.tolist() == [101,102,103,104]
    assert merged.educ3.fillna('M').tolist() == ['1','2','3','M']
    assert p1.shape == (4,7) and p2.shape == (4,2)
    assert_raises(ValueError, merge_checked, p1, p2.assign(id=[103,101,101,102]))
    assert_raises(ValueError, merge_checked, p1, p2.iloc[:-1].copy())
    assert_raises(ValueError, merge_checked, p1, pd.concat([p2,pd.DataFrame({'id':[900], 'educ3':['3']})]))
    assert_raises(ValueError, audit_codes, merged, {'1','2'})
    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        good = tmp/'p2.csv'
        p2.to_csv(good,index=False)
        reread = read_p2(good)
        pd.testing.assert_frame_equal(reread,p2,check_dtype=False)
        schema = tmp/'schema.json'
        schema.write_text(json.dumps({'value_labels':{'1':'Low','2':'Middle','3':'High'},
                                     'country_rules_verified':True,
                                     'method_source':'SYNTHETIC TEST ONLY'}), encoding='utf-8')
        allowed, _ = read_schema(schema)
        assert allowed == {'1','2','3'}
        schema.write_text(json.dumps({'value_labels':{'1':'Low','2':'Middle','3':'High'},
                                     'country_rules_verified':False,
                                     'method_source':'SYNTHETIC TEST ONLY'}), encoding='utf-8')
        assert_raises(ValueError, read_schema, schema)
        badp2 = tmp/'bad.csv'
        badp2.write_text('id,educ3\n101,1\n101,2\n',encoding='utf-8')
        assert_raises(ValueError, read_p2, badp2)
    assert_raises(ValueError, export_atomic, merged, ROOT/'data/raw/UNSAFE.csv')
    print('PASS: respondent-order-independent one-to-one merge; all original columns unchanged')
    print('PASS: duplicate/missing/extra P2 IDs detected; undocumented educ3 code detected')
    print('PASS: P2 schema must document labels, provenance and reviewed country rules')
    print('PASS: unsafe output paths rejected; no data exported by self-test')
    print('TASK 30 SELF-TEST PASSED (synthetic P2 values only; real mapping remains pending)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--p2', type=Path, default=None, help='P2 CSV containing id,educ3 for ALL 28,333 IDs')
    ap.add_argument('--schema', type=Path, default=None,
                    help='P2-authored JSON value_labels, method_source, country_rules_verified')
    ap.add_argument('--export-final', action='store_true', help='Opt-in export ONLY with reviewed P2 schema')
    ap.add_argument('--overwrite', action='store_true', help='Allow replacing existing DIFFERENT final file')
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.overwrite and not args.export_final:
            ap.error('--overwrite is only valid with --export-final')
        if args.preflight and args.export_final:
            ap.error('--preflight and --export-final cannot be used together')
        run(preflight=args.preflight, p2_path=args.p2, schema_path=args.schema,
            export_final=args.export_final, overwrite=args.overwrite)
