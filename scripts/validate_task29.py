"""COMP5122M Coursework 1, Task 29 — export and privately share provisional P1 data.

Authoritative (original SAV + CSV; creates private provisional dataset):
    python scripts/validate_task29.py --self-test
    python scripts/validate_task29.py

Optional CSV/dictionary-proxy diagnostic only (NEVER exports a data file):
    python scripts/validate_task29.py --csv-diagnostic --labels-json PATH_TO_LABELS_JSON

The provisional output is data/clean/pew_clean_p1_provisional.csv, NEVER staged
for Git. Share it manually to the authorised teammates via university Teams.
It intentionally retains raw d_educ_* fields until P2 sends the educ3 mapping.
No source file or existing data/clean/pew_clean.csv is overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
try:
    from src.cleaning import do_clean, selected_columns, missing_codes
    from validate_task18 import independent_age_gender
    from validate_task28 import assert_equal_by_id
except ImportError as exc:
    raise SystemExit('Task 29 requires src/cleaning.py and the Task 18/28 '
                     f'validation scripts in your local repository: {exc}') from exc

SAV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = ROOT / 'data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
OUT = ROOT / 'data/clean'
EXPORT = OUT / 'pew_clean_p1_provisional.csv'
N = 28_333
BANDS = {'18–29': 6068, '30–49': 9787, '50–64': 6596, '65+': 5697}
EXPECTED_RECODE = {'ai_heard': 808, 'ai_cncexc': 1964, 'aireg_us': 362,
                   'aireg_china': 380, 'aireg_eu': 581, 'aireg_country': 213,
                   'internetfreq': 94, 'age': 294}


def numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='raise')
    s = series.astype('string').str.strip()
    return pd.to_numeric(s.mask(s.eq('')), errors='raise')


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ensure_private_path(path: Path) -> None:
    """Refuse data export outside data/clean, even if user supplies odd filenames."""
    assert path.parent.resolve() == OUT.resolve(), 'Export must be directly under data/clean/'
    assert path.name == EXPORT.name, 'Unexpected provisional dataset filename'


def preflight_git_protection() -> str:
    """Require actual ignore protection; detect previously tracked data.

    If Git is not initialized in a diagnostic environment, verify a conservative
    local ignore declaration. On a real Git checkout use `git check-ignore`.
    """
    gi = ROOT / '.gitignore'
    if not gi.is_file():
        raise RuntimeError('Missing .gitignore: install the supplied .gitignore '
                           'or add /data/raw/ and /data/clean/ to your existing one.')
    rules = gi.read_text(encoding='utf-8')
    active = [row.strip().lstrip('/') for row in rules.splitlines()
              if row.strip() and not row.strip().startswith(('#', '!'))]
    raw_rule = any(rule.startswith('data/raw/') for rule in active)
    clean_rule = any(rule.startswith('data/clean/') for rule in active)
    if not (raw_rule and clean_rule):
        raise RuntimeError('Your .gitignore does not clearly exclude both /data/raw/ and /data/clean/. '
                           'Add both protections before exporting.')

    git = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--is-inside-work-tree'],
                         capture_output=True, text=True, check=False)
    if git.returncode != 0 or git.stdout.strip() != 'true':
        print('NOTE: Git repository not visible here; .gitignore rules were inspected, '
              'but Git tracking must be checked in your actual clone.')
        return 'ignore_patterns_checked;git_unavailable'

    for rel in ['data/raw/example.sav', 'data/clean/pew_clean_p1_provisional.csv']:
        ignored = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '--no-index', '-q', rel],
                                 capture_output=True, text=True, check=False)
        if ignored.returncode != 0:
            raise RuntimeError(f'Git would NOT ignore {rel}. Fix your .gitignore before export.')

    tracked = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '--', 'data/raw/', 'data/clean/'],
                             capture_output=True, text=True, check=True)
    unsafe = [line for line in tracked.stdout.splitlines()
              if not line.endswith('/.gitkeep') and not line.endswith('/.gitignore')]
    if unsafe:
        raise RuntimeError('Private data already tracked by Git! Remove from index with '
                           '`git rm --cached <filename>`; do not push. Tracked: ' + ', '.join(unsafe[:8]))
    print('PASS: Git ignores raw and cleaned data; no respondent-level files tracked')
    return 'git_ignore_verified;no_private_files_tracked'


def verify_clean(raw: pd.DataFrame, clean: pd.DataFrame, meta: object,
                 *, source: str) -> None:
    assert raw.shape == (N, 483), f'{source}: unexpected source shape {raw.shape}'
    assert len(clean) == N and raw.id.is_unique and clean.id.is_unique
    assert clean.id.equals(raw.id), f'{source}: respondent order/IDs changed'
    assert len(clean.columns) == len(selected_columns(raw)) + 6 == 70, (
        f'{source}: expected 70 Part I fields; found {len(clean.columns)}')
    assert 'educ3' not in clean, f'{source}: educ3 is not available until P2 sends mapping'
    assert int(clean.age_band.isna().sum()) == 185
    assert int(clean.age_band_recovered.fillna(False).sum()) == 111
    for band, expected in BANDS.items():
        assert int(clean.age_band.eq(band).sum()) == expected, f'{source}: {band} count differs'
    source_totals = clean.age_band_source.value_counts(dropna=False).to_dict()
    assert source_totals == {'exact_age':26210, 'australia_group':1827,
                             'followup_compatible':111, 'unknown':185}, (
        f'{source}: age provenance counts differ: {source_totals}')
    assert int(clean.gender_clean.eq(1).sum()) == 15107
    assert int(clean.gender_clean.eq(2).sum()) == 13203
    assert int(clean.gender_clean.eq(3).sum()) == 21
    assert int(clean.gender_clean.isna().sum()) == 2
    assert int(clean.internetfreq.eq(6).sum()) == 3303
    assert int(clean.age.eq(97).sum()) == 3
    for col, count in EXPECTED_RECODE.items():
        original, cleaned = numeric(raw[col]), numeric(clean[col])
        bad = missing_codes(meta, col)
        assert bad, f'{source}: source labels did not identify {col} nonresponses'
        mask = original.isin(bad)
        assert int(mask.sum()) == count, f'{source}: {col} unexpected recode count'
        expected = original.mask(mask).to_numpy(dtype=float, na_value=np.nan)
        actual = cleaned.to_numpy(dtype=float, na_value=np.nan)
        assert np.isclose(expected, actual, rtol=0, atol=0, equal_nan=True).all(), (
            f'{source}: {col} changed beyond labelled nonresponses')
    for col, count in [('aireg_us', 3112), ('aireg_china', 3376),
                       ('aireg_eu', 4019), ('aireg_country', 2946)]:
        assert int(clean[col].eq(8).sum()) == count, f'{source}: lost valid Q32 Not sure'
    for col in [c for c in clean if c.startswith('d_educ_')]:
        x = numeric(raw[col]).to_numpy(dtype=float, na_value=np.nan)
        y = numeric(clean[col]).to_numpy(dtype=float, na_value=np.nan)
        assert np.isclose(x, y, rtol=0, atol=0, equal_nan=True).all(), (
            f'{source}: raw P2 education {col} was altered')
    assert (clean.weight.notna() & clean.weight.gt(0)).all()
    assert np.isclose(numeric(raw.weight), numeric(clean.weight),
                      rtol=0, atol=0).all(), f'{source}: survey weight altered'
    print(f'PASS: {source}: 28,333 respondents, 70 output columns, 111 age recoveries, '
          '185 missing age bands, 21 Other genders, source education unchanged')


def read_originals(*, diagnostic: bool, labels_json: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    if not CSV.is_file():
        raise FileNotFoundError('Missing original CSV at data/raw/. Use the original exact filename.')
    csv = pd.read_csv(CSV, encoding='utf-8-sig', low_memory=False)
    csv.columns = csv.columns.str.strip().str.lower()
    if diagnostic:
        if labels_json is None or not labels_json.is_file():
            raise FileNotFoundError('CSV diagnostic requires --labels-json FILE from the original dictionary')
        label_map = json.loads(labels_json.read_text(encoding='utf-8'))
        meta = SimpleNamespace(variable_value_labels={k:{float(x):str(v) for x,v in lab.items()}
                                                       for k,lab in label_map.items()})
        return csv.copy(deep=True), csv, meta
    if not SAV.is_file():
        raise FileNotFoundError('Missing original SAV at data/raw/. Add the original SAV file.')
    try:
        import pyreadstat
    except ImportError as exc:
        raise RuntimeError('Install pyreadstat in your activated venv: pip install pyreadstat') from exc
    sav, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    sav.columns = sav.columns.str.strip().str.lower()
    return sav, csv, meta


def write_if_safe(clean: pd.DataFrame, *, overwrite: bool) -> tuple[str, str]:
    """Atomic publish to local-only data/clean; compare if file already present."""
    ensure_private_path(EXPORT)
    OUT.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix='.task29_export_', suffix='.tmp', dir=OUT)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        clean.to_csv(tmp, index=False, encoding='utf-8')
        # Ensure serialized CSV still contains precisely the verified output.
        roundtrip = pd.read_csv(tmp, low_memory=False, encoding='utf-8-sig')
        assert_equal_by_id(clean, roundtrip, list(clean.columns), stage='Provisional CSV roundtrip')
        digest = fingerprint(tmp)
        if EXPORT.is_file() and not overwrite:
            prior_hash = fingerprint(EXPORT)
            if prior_hash != digest:
                raise RuntimeError('Existing provisional file has DIFFERENT contents. '
                                   'Refusing overwrite. Inspect it first; use --overwrite only '
                                   'after confirming an intentional replacement.')
            print('PASS: existing provisional file already matches exactly; no overwrite needed')
            return prior_hash, 'previous_matching_export'
        os.replace(tmp, EXPORT)
        print('PASS: provisional P1 CSV written to private data/clean/ (not to Git)')
        return digest, 'new_private_export'
    finally:
        tmp.unlink(missing_ok=True)


def output_evidence(clean: pd.DataFrame, digest: str, git_status: str,
                    export_status: str) -> None:
    from datetime import datetime, timezone
    OUT.mkdir(parents=True, exist_ok=True)
    size = EXPORT.stat().st_size
    summary = pd.DataFrame([{
        'task': 29, 'status': 'VALIDATED_SAV_PENDING_PRIVATE_TEAM_UPLOAD',
        'source': 'original_SAV_metadata_and_original_CSV',
        'export_filename': EXPORT.name, 'export_bytes': size, 'sha256': digest,
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'rows': len(clean), 'columns': len(clean.columns),
        'known_age_bands': int(clean.age_band.notna().sum()),
        'missing_age_bands': int(clean.age_band.isna().sum()),
        'recovered_age_bands': int(clean.age_band_recovered.sum()),
        'other_gender_preserved': int(clean.gender_clean.eq(3).sum()),
        'educ3_integrated': False, 'education_mapping_status': 'pending_P2',
        'git_protection': git_status, 'export_status': export_status,
        'private_upload_confirmed': False,
    }])
    summary.to_csv(OUT/'task29_provisional_export_manifest.csv',index=False)
    pd.DataFrame([{'task':29, 'decision':'Share P1 provisional cleaned microdata privately',
                   'alternative':'Commit microdata or manually merge education early',
                   'reason':'Private handoff; prevents accidental disclosure and avoids unsupported P2 mapping',
                   'effect':'28,333 rows and 70 columns validated; 111 age recoveries; 185 missing age bands',
                   'status':'Export verified; team upload pending'}]).to_csv(
                       OUT/'task29_decision_log_entry.csv',index=False)
    pd.DataFrame([{'task':29, 'ai_contribution':'Generated safe P1 provisional CSV export and validation',
                   'verification':'SAV and CSV by ID; age/gender independently reconstructed; CSV roundtrip; Git ignore checked',
                   'human_step':'Confirm file uploaded to restricted Teams and P2 has it; no automated upload attempted',
                   'status':'SAV export verified; private sharing not yet confirmed'}]).to_csv(
                       OUT/'task29_ai_use_log_entry.csv',index=False)
    pd.DataFrame([{'task':29,
                   'ai_error':'First Task 29 diagnostic attempted a numeric comparison of all 483 raw columns, including date strings',
                   'how_detected':'CSV diagnostic raised ValueError on date string 2/10/2025',
                   'root_cause':'Reused numeric-only respondent comparator on non-numeric original metadata fields',
                   'correction':'Compare all 64 selected Part I source fields with strict numeric codes; independently compare all 70 cleaned fields. Prior Task 5 already audited the wider raw SAV/CSV.',
                   'verification':'Repeat CSV diagnostic passed across all 28,333 respondents; no source values modified',
                   'status':'CORRECTED_BEFORE_STUDENT_RELEASE'}]).to_csv(
                       OUT/'task29_ai_discrepancy_entry.csv',index=False)
    (OUT/'task29_private_handoff_note.txt').write_text(
        'Task 29 — P1 PROVISIONAL DATASET, NOT FINAL\n'
        f'Local private file: data/clean/{EXPORT.name}\n'
        f'Rows: {len(clean):,}; columns: {len(clean.columns)}; SHA-256: {digest}\n'
        'Validation: source SAV and CSV match by respondent ID; independent age/gender check; CSV roundtrip.\n'
        'Known age bands: 28,148; missing age bands: 185; 111 safely recovered.\n'
        'Q32 Not sure and Australian Other gender intentionally retained.\n'
        'Original d_educ_* education columns are retained and unharmonised. P2 educ3 pending.\n'
        'Five unknown French income code 98 values remain flagged in Task 28 audit (income not exported).\n'
        'PRIVATE SHARING (person must carry out):\n'
        '1. Open university Teams; select the intended restricted course group/chat.\n'
        '2. Upload data/clean/pew_clean_p1_provisional.csv to that group ONLY.\n'
        '3. Tell teammates this is a provisional P1 microdata handoff; P2 educ3 is NOT integrated.\n'
        '4. Do NOT commit, attach to a public issue, or publish the CSV or respondent-level files.\n'
        '5. Once P2 sends id,educ3 mapping, separately verify the merge and create the FINAL dataset.\n',
        encoding='utf-8')
    print('Saved: task29_provisional_export_manifest.csv, task29_private_handoff_note.txt,')
    print('       task29_decision_log_entry.csv, task29_ai_use_log_entry.csv,')
    print('       task29_ai_discrepancy_entry.csv')


def run(*, diagnostic: bool, labels_json: Path | None, overwrite: bool) -> None:
    print('=== TASK 29: PRIVATE P1 PROVISIONAL DATASET EXPORT ===')
    print('Mode:', 'CSV dictionary-proxy DIAGNOSTIC (NO EXPORT)' if diagnostic else 'ORIGINAL SAV + CSV')
    git_status = preflight_git_protection()
    raw, csv, meta = read_originals(diagnostic=diagnostic, labels_json=labels_json)
    assert_equal_by_id(raw, csv, selected_columns(raw), stage='Original SAV vs CSV, all 64 Part I fields')
    before = raw[['id','country','weight']].copy(deep=True)
    baseline = csv[['id','country','weight']].copy(deep=True)
    sav_clean = do_clean(raw, meta, [])
    csv_clean = do_clean(csv, meta, [])
    verify_clean(raw, sav_clean, meta, source='SAV' if not diagnostic else 'CSV-proxy')
    verify_clean(csv, csv_clean, meta, source='companion CSV')
    assert_equal_by_id(sav_clean, csv_clean, list(sav_clean.columns),
                       stage='Cleaned SAV vs CSV, ALL 70 columns')
    independently_calculated = independent_age_gender(raw, meta)
    assert_equal_by_id(sav_clean, independently_calculated,
                       ['age_band', 'gender_clean', 'gender_label'],
                       stage='Independent Task 14/16/17 age and gender reconstructions')
    assert_equal_by_id(raw, before, ['country','weight'],stage='Source raw dataset unchanged')
    assert_equal_by_id(csv, baseline, ['country','weight'],stage='Source CSV dataset unchanged')

    if diagnostic:
        print('TASK 29 CSV DIAGNOSTIC PASSED: 28,333 rows, 70 columns; no dataset exported.')
        print('Run without --csv-diagnostic on your PC to validate SAV and export a provisional CSV.')
        return
    digest, export_status = write_if_safe(sav_clean, overwrite=overwrite)
    output_evidence(sav_clean, digest, git_status, export_status)
    print('SHA-256:', digest)
    print('WARNING: Export is PROVISIONAL — waiting for P2 education harmonisation.')
    print('WARNING: PRIVATE TEAMS UPLOAD NOT YET DONE; upload manually to authorised classmates.')
    print('TASK 29 EXPORT PASSED: SAV/CSV and independent checks; 28,333 rows, 70 columns;')
    print('                      private local CSV generated, no Git tracking or public upload.')


def self_test() -> None:
    from pathlib import PurePosixPath
    assert PurePosixPath('data/clean/pew_clean_p1_provisional.csv').parts[:2] == ('data','clean')
    ensure_private_path(EXPORT)
    try:
        ensure_private_path(ROOT/'data/raw/pew_clean_p1_provisional.csv')
    except AssertionError:
        pass
    else:
        raise AssertionError('Unsafe export path was accepted')
    synthetic = pd.DataFrame({'id':[21,22,23], 'weight':[1.0,2.5,3.0],
                              'age_band':['18–29','65+',None],
                              'age_band_recovered':[False,True,False],
                              'gender_label':['Male','Other',None]})
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)/'out.csv'
        synthetic.to_csv(p,index=False)
        loaded = pd.read_csv(p)
        assert_equal_by_id(synthetic, loaded, list(synthetic.columns),
                           stage='Self-test CSV roundtrip')
        assert len(fingerprint(p)) == 64
        corrupted = loaded.copy()
        corrupted.loc[0,'weight'] = 99
        try:
            assert_equal_by_id(synthetic, corrupted, ['weight'],stage='Self-test deliberate corruption')
        except AssertionError:
            pass
        else:
            raise AssertionError('Deliberate CSV corruption went undetected')
    print('PASS: safe export location; unsafe paths rejected')
    print('PASS: CSV numeric/string/Boolean roundtrip and SHA-256 manifest hashing')
    print('PASS: altered respondent data detected')
    print('TASK 29 SELF-TEST PASSED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--csv-diagnostic', action='store_true', help='Requires --labels-json and never exports microdata')
    parser.add_argument('--labels-json', type=Path, default=None)
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing DIFFERENT provisional export after deliberate review')
    opts = parser.parse_args()
    if opts.self_test:
        self_test()
    else:
        run(diagnostic=opts.csv_diagnostic, labels_json=opts.labels_json, overwrite=opts.overwrite)
