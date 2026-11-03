"""Task 35: consolidate and audit AI assistance and genuine AI discrepancies.

Run from the COMP5122M_cw1 repository root:
    python scripts/validate_task35.py --self-test
    python scripts/validate_task35.py

TASK 35 IS A DOCUMENTATION AUDIT, NOT A RE-EXECUTION OF DATA CLEANING.
- Reads existing per-task AI-use and AI-discrepancy CSVs, preserving their wording.
- Tasks 1–12 are explicitly flagged as retrospective summaries from prior
  conversation/validator history; they are NOT falsely presented as source CSVs.
- Task 30 remains pending P2; Task 29's Teams upload remains pending.
- Known Task 18 import error is a retrospective, chat-reported entry (not
  silently represented as an original per-task discrepancy CSV).
- A methodological limitation alone is NOT automatically an AI coding error.
- Final mode FAILS if any required locally produced log is missing.
- --local-diagnostic reports missing source files but cannot pass final checks.
- Never reads, rewrites or uploads individual survey records or microdata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'clean'
SOURCE_AI_TASKS = tuple(list(range(13, 30)) + list(range(31, 35)))
EXPECTED_ERROR_FILE_TASKS = (18, 21, 23, 27, 28, 29, 32)

AI_COLUMNS = [
    'task', 'topic', 'ai_contribution', 'source_evidence', 'source_record_original_status',
    'verification_recorded_in_source', 'verification_by_student_report',
    'scope_or_known_limitations', 'evidence_kind', 'review_required',
]
ERROR_COLUMNS = [
    'issue_id', 'task', 'issue_type', 'original_ai_claim_or_error',
    'how_detected', 'correction_or_clarification', 'verification_after_correction',
    'effect_on_source_data', 'source_status_original', 'source_evidence',
    'human_review_required', 'source_record_json',
]

# Short, explicitly RETROSPECTIVE descriptions of tool assistance, backed by
# earlier validation scripts and their output in this coursework conversation.
# They are not transcribed CSV rows, and must be reviewed by the student.
EARLY_AI = {
    1: ('Repository setup', 'Assisted with project structure, branch and keeping survey microdata private.'),
    2: ('Source inspection', 'Created source loading and dimension/country validation guidance.'),
    3: ('Field selection', 'Generated checks for 64 Part I columns while preserving all respondents.'),
    4: ('SAV metadata', 'Generated label-based nonresponse audit and codebook exception review.'),
    5: ('Source equivalence', 'Generated respondent-level SAV-versus-CSV comparison for all 483 fields.'),
    6: ('Duplicates', 'Generated duplicate respondent ID and row assessments.'),
    7: ('Metadata exceptions', 'Audited observed response codes against the dictionary; flagged French income 98 without guessing.'),
    8: ('Country routing', 'Generated checks for country-specific age and education fields.'),
    9: ('Survey weights', 'Generated individual-weight and country-sum checks.'),
    10: ('Q30/Q31 recoding', 'Generated label-driven Q30/Q31 nonresponse validation.'),
    11: ('Q32 recoding', 'Generated refusal-only recoding checks; preserved valid Not sure.'),
    12: ('Internet frequency', 'Generated nonresponse/routing checks while retaining legitimate non-users.'),
}

# This was observed by the student in the conversation but does not have an
# original Task 18 ai_discrepancy_entry.csv row. Keep its evidence distinct.
CHAT_REPORTED_ERRORS = [
    {
        'task': 18, 'issue_type': 'AI-generated dependency/setup failure — retrospective chat report',
        'original_ai_claim_or_error': 'Task 18 validator initially could not import src.cleaning (ModuleNotFoundError).',
        'how_detected': 'Student PowerShell traceback from python scripts/validate_task18.py.',
        'correction_or_clarification': 'Provide/restore src/cleaning.py and ensure repository module import works.',
        'verification_after_correction': 'Student subsequently reported TASK 18 PASSED with original SAV and CSV.',
        'effect_on_source_data': 'None; initial validation aborted before source transformations.',
        'source_status_original': 'RETROSPECTIVE_FROM_CONVERSATION',
        'source_evidence': 'Conversation: Task 18 traceback and subsequent successful rerun; review alongside local terminal history.',
        'source_record_json': '',
    },
]


# Errors found by RUNNING or CHECKING the AI output (P1 Cleaning Tasks doc, each
# task's "AI log"). ChatGPT's own discrepancy files only hold its self-reports,
# so without these rows the log would miss most of the real errors.
DOC = 'P1 Cleaning Tasks doc, Task {t} AI log; corrected script rerun PASSED'
DOC_VERIFIED_ERRORS = [
 (2, 'ChatGPT', 'First Task 2 script never called main() (ran silently), checked the CSV against sizes copied from the CSV, and claimed a questionnaire check it did not do',
  'Running it gave no output; reading the code', 'main() called; independent sizes; false claim removed'),
 (4, 'ChatGPT', 'Script supplied in pieces with its middle missing, so it ran with no output',
  'Running it', 'Replaced by a complete tested script'),
 (4, 'Claude', 'Claude first flagged household-size numbers as stray missing codes',
  'Checking flagged columns against labels', 'Fixed before use'),
 (5, 'ChatGPT', 'SAV/CSV comparison checked only 17 of 64 Part I columns and hid the CSV blank-space issue',
  'Comparing the script scope with the task', 'Full 64- and 483-column comparison that records the blanks'),
 (6, 'ChatGPT', 'Strongest duplicate check still compared weights and interview dates, which would hide a copied interview',
  'Reviewing what each check compares', 'Added an answers-only duplicate check (still 0)'),
 (8, 'ChatGPT', 'Routing check covered only Australian columns and missed 50 other country-specific Part I columns',
  'Comparing with the 64-column list', 'Added check; all 50 pass, 2 Israeli answers flagged'),
 (9, 'ChatGPT', 'Code comment said the 0.06-5.89 weight range came from the brief; it came from our own data check',
  'Checking the brief', 'Comment corrected; brief not cited for it'),
 (10, 'ChatGPT', 'Task 10 and 11 scripts imported functions from ChatGPT\'s own Task 4 file, not ours, so would crash',
  'Running it', 'Functions added to our Task 4 file'),
 (12, 'ChatGPT', 'Said its script ran successfully, but it stopped on real data: "SAV and CSV IDs differ" (float vs integer IDs)',
  'Running it on the real files', 'Both IDs cast to the same type (line 85); all counts then matched'),
 (24, 'ChatGPT', 'Don\'t-know sensitivity tested Q30 only (0.5 pp), although Task 23 showed DK varies with age mainly on Q31',
  'Comparing with Task 23', 'Added validate_task24_q31.py: changes up to 28.6 pp'),
 (27, 'ChatGPT', 'Assumed cleaning.py drop_other_gender leaves gaps unchanged; it removes the 21 people, so the script crashed',
  'Running it', 'Assertion checks against the gender-recorded version; only Australia changes'),
 (29, 'ChatGPT', 'Report audit table had no effect-on-results column and left out 5 known issues',
  'Comparing with earlier verified tasks', 'Claude added them in the report version'),
 (31, 'ChatGPT', 'Notebook crashed with KeyError gender_other_to_missing and expected 0 for a test that removes 21 people',
  'Running it in a fresh kernel', 'Name exclude_australia_other_gender, expected 0.044270; all 7 cells pass'),
 (32, 'ChatGPT', 'Concluded the checklist age figure of 0.22 was unsupported, using the pooled V of 0.085',
  'Claude computed V within each country: age about 0.20, internet about 0.27', 'Report pooled 0.085 and within-country 0.20; 0.22 was not an error'),
 (34, 'ChatGPT', 'Hand-typed Task 8 summary said "19 routing checks passed" and Task 32 row repeated the misleading age conclusion',
  'Comparing rows with the doc', 'Both rows fixed in the script before the run'),
 (0, 'Claude', 'Claude said P2/P3 would analyse Q32 by age (that is Part II) and that a ChatGPT script was cut off (only part had been pasted)',
  'User questioned it / full file checked', 'Corrected in chat and doc'),
]


def doc_errors() -> list[dict]:
    out = []
    for i, (t, who, err, how, fix) in enumerate(DOC_VERIFIED_ERRORS, start=1):
        out.append({'issue_id': f'T{t:02d}-DOC-{i:02d}', 'task': t,
            'issue_type': f'{who} error found by running/checking (not self-reported)',
            'original_ai_claim_or_error': err, 'how_detected': how,
            'correction_or_clarification': fix,
            'verification_after_correction': DOC.format(t=t if t else 'various'),
            'effect_on_source_data': 'None; caught before any result was used',
            'source_status_original': 'CORRECTED', 'source_evidence': DOC.format(t=t if t else 'various'),
            'human_review_required': 'Check wording against the doc.', 'source_record_json': ''})
    return out


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 18), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def available(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().casefold() not in ('', 'nan', 'none', 'null', '<na>')


def first(row: dict, *keys: str) -> str:
    for key in keys:
        if key in row and available(row[key]):
            return str(row[key]).strip()
    return ''


def source_path(root: Path, task: int, kind: str, diagnostic: bool) -> Path | None:
    actual = root / f'task{task:02d}_{kind}_entry.csv'
    if actual.is_file():
        return actual
    if diagnostic:
        fallback = root / f'task{task:02d}_csv_diagnostic_{kind}_entry.csv'
        if fallback.is_file():
            return fallback
    return None


def read_records(source: Path, task: int) -> list[dict]:
    data = pd.read_csv(source, dtype=str, keep_default_na=False)
    if data.empty:
        raise AssertionError(f'{source.name}: expected nonempty evidence file')
    records = data.to_dict(orient='records')
    for row in records:
        recorded = first(row, 'task')
        if recorded and recorded not in (str(task), f'{task:02d}'):
            raise AssertionError(f'{source.name}: task value {recorded!r} differs from {task}')
    return records


def verify_source_entry(task: int, source: Path) -> dict:
    records = read_records(source, task)
    if len(records) != 1:
        raise AssertionError(f'{source.name}: expected one AI-use entry, found {len(records)}')
    row = records[0]
    contribution = first(row, 'ai_contribution', 'ai_use', 'contribution')
    if not contribution:
        raise AssertionError(f'{source.name}: no documented AI contribution')
    validation = first(row, 'verification', 'independent_evidence',
        'independent_reference', 'independent_checks', 'human_validation',
        'validation', 'evidence', 'reference', 'source', 'result')
    student = first(row, 'human_verification', 'human_validation', 'verification_outcome', 'result')
    source_status = first(row, 'status', 'verification_outcome', 'human_verification', 'validation') or 'NOT_RECORDED'
    caution = first(row, 'limitation', 'limitations', 'ai_error_or_limitation', 'ai_error_or_clarification',
                    'ai_errors_identified', 'ai_errors', 'ai_error', 'ai_discrepancy', 'ai_error_review')
    # Deliberately do not change stale source wording such as 'pending student SAV'.
    return {'task': task, 'topic': f'Task {task} — AI assistance',
        'ai_contribution': contribution,
        'source_evidence': f'{source.name} [sha256:{digest(source)}]',
        'source_record_original_status': source_status,
        'verification_recorded_in_source': validation or 'See original validation output.',
        'verification_by_student_report': ('Original SAV/CSV validation or task execution reported PASSED in conversation; '
            'source entry may retain older pending wording. Review local terminal output.'),
        'scope_or_known_limitations': caution or 'No particular AI error identified in this source entry.',
        'evidence_kind': 'CSV_DIAGNOSTIC_ONLY__NOT_FINAL' if 'csv_diagnostic' in source.name else 'ORIGINAL_PER_TASK_AI_USE_CSV',
        'review_required': 'Check source record and actual terminal result before submission.'}


def make_ai_log(root: Path, *, diagnostic: bool) -> pd.DataFrame:
    records = []
    for task, (topic, content) in EARLY_AI.items():
        records.append({'task':task, 'topic':topic, 'ai_contribution':content,
            'source_evidence':f'RETROSPECTIVE: earlier Task {task} validation conversation/scripts; no dedicated AI-use CSV',
            'source_record_original_status':'NO_ORIGINAL_TASK_AI_USE_ENTRY',
            'verification_recorded_in_source':'Prior SAV/CSV validation reported by student in conversation',
            'verification_by_student_report':'PASSED_REPORTED__REVIEW_EARLY_TERMINAL_HISTORY',
            'scope_or_known_limitations':'Retrospective wording; not a verbatim extraction of a saved AI-use log.',
            'evidence_kind':'RETROSPECTIVE__HUMAN_REVIEW_REQUIRED',
            'review_required':'Review against Tasks 1–12 terminal outputs and coursework AI-use policy.'})
    for task in SOURCE_AI_TASKS:
        path = source_path(root, task, 'ai_use_log', diagnostic)
        if path is None:
            if not diagnostic:
                raise FileNotFoundError(f'Task {task} AI-use entry missing: '
                    f'{root / f"task{task:02d}_ai_use_log_entry.csv"}. '
                    'Run the corresponding task validator; do not invent a replacement.')
            records.append({'task':task, 'topic':f'Task {task} — MISSING SOURCE',
                'ai_contribution':'NOT ESTABLISHED FROM THIS PARTIAL SNAPSHOT',
                'source_evidence':'MISSING_EVIDENCE',
                'source_record_original_status':'MISSING_EVIDENCE',
                'verification_recorded_in_source':'NOT VERIFIED',
                'verification_by_student_report':'NOT VERIFIED FROM FILES',
                'scope_or_known_limitations':'This row cannot be submitted as verified.',
                'evidence_kind':'MISSING_EVIDENCE__DO_NOT_SUBMIT',
                'review_required':'Find original per-task file on student machine.'})
        else:
            records.append(verify_source_entry(task, path))
    # Do not treat 'ran preflight' as 'P2 merged'. No real merge is known.
    records.append({'task':30, 'topic':'P2 education integration — pending',
        'ai_contribution':'Generated an education id,educ3 merge validator and strict preflight/schema template.',
        'source_evidence':'validate_task30.py; student preflight reported; actual P2 CSV not provided',
        'source_record_original_status':'NO_COMPLETED_P2_MERGE',
        'verification_recorded_in_source':'Task 30 preflight/checksum reported; real merge not executed',
        'verification_by_student_report':'PENDING_P2__NOT_COMPLETED',
        'scope_or_known_limitations':'Do not invent educ3 categories or assert final dataset exists.',
        'evidence_kind':'RETROSPECTIVE_PENDING_DEPENDENCY',
        'review_required':'Update only after P2 mapping is received and real join passes.'})
    records.append({'task':35, 'topic':'Consolidate AI use and error provenance',
        'ai_contribution':'Generated a strict, source-preserving AI-use/error consolidation and test script.',
        'source_evidence':'validate_task35.py (this run); Task34 consolidated decision log',
        'source_record_original_status':'CURRENT_RUN',
        'verification_recorded_in_source':'Script self-test and local file-inventory checks',
        'verification_by_student_report':'PENDING_USER_RUN_AND_LOG_REVIEW',
        'scope_or_known_limitations':'Log completeness is not proof that the student has inspected every claim.',
        'evidence_kind':'CURRENT_AI_GENERATED_ENTRY__AWAITING_REVIEW',
        'review_required':'Read consolidated CSV and correct retrospective text as needed.'})
    return pd.DataFrame(records, columns=AI_COLUMNS).sort_values('task').reset_index(drop=True)


def normalize_error(task: int, row: dict, source: Path, ordinal: int) -> dict:
    original = first(row,'original_ai_error','original_ai_or_checklist_claim','original_claim',
        'original_issue','ai_error','error','earlier_ai_claim','issue')
    corrected = first(row,'correction','resolution','corrected_finding','fix')
    if not original or not corrected:
        raise AssertionError(f'{source.name} row {ordinal}: missing original problem/correction; cannot claim this error is documented')
    detection = first(row,'how_detected','detection','detected_by','verification',
        'why','discrepancy') or 'Detection method omitted from source; requires student review.'
    validation = first(row,'validation','verification','status') or 'Confirm corrected task execution in terminal logs.'
    effect = first(row,'effect_on_data','effect_on_original_data','data_effect')
    if not effect:
        effect = 'Not separately described in source: check whether any dataset was affected.'
    original_status = first(row,'status') or 'NOT_STATED_IN_SOURCE'
    return {'issue_id':f'T{task:02d}-{ordinal:02d}', 'task':task,
        'issue_type':'SOURCE_RECORDED_AI_ERROR_OR_CORRECTION',
        'original_ai_claim_or_error':original, 'how_detected':detection,
        'correction_or_clarification':corrected,
        'verification_after_correction':validation,
        'effect_on_source_data':effect,
        'source_status_original':original_status,
        'source_evidence':f'{source.name} [sha256:{digest(source)}] row {ordinal}',
        'human_review_required':'Review original source; source status may predate student successful rerun.',
        'source_record_json':json.dumps(row,ensure_ascii=False,sort_keys=True)}


def make_error_log(root: Path, *, diagnostic: bool) -> tuple[pd.DataFrame,pd.DataFrame]:
    records=[]
    source_checks=[]
    for task in EXPECTED_ERROR_FILE_TASKS:
        path = source_path(root,task,'ai_discrepancy',diagnostic)
        found=path is not None
        source_checks.append({'task':task,'expected_file':f'task{task:02d}_ai_discrepancy_entry.csv',
            'found':found,'source_name':path.name if found else 'MISSING_EVIDENCE',
            'kind':'ORIGINAL' if found and 'csv_diagnostic' not in path.name else 'CSV_DIAGNOSTIC_ONLY' if found else 'MISSING'})
        if not found:
            if not diagnostic:
                raise FileNotFoundError(f'Task {task} AI discrepancy source missing: '
                    f'{root / f"task{task:02d}_ai_discrepancy_entry.csv"}. '
                    'Do not silently replace it with a summary.')
            continue
        for idx,row in enumerate(read_records(path,task), start=1):
            records.append(normalize_error(task,row,path,idx))
    for idx, row in enumerate(CHAT_REPORTED_ERRORS, start=1):
        records.append({'issue_id':f'T{row["task"]:02d}-CHAT-{idx:02d}', 'task':row['task'],
            **{k:row[k] for k in ERROR_COLUMNS if k in row and k not in ('issue_id','task')},
            'human_review_required':'RETROSPECTIVE: verify traceback and successful rerun in chat/local terminal.'})
    if True:
        records.extend(doc_errors())
    for r in records:
        if r['task'] == 32 and '0.22' in r['original_ai_claim_or_error'] and 'DOC' not in r['issue_id']:
            r['issue_type'] = 'NOT AN ERROR: superseded; 0.22 matches within-country V (about 0.20), see T32-DOC row'
    return (pd.DataFrame(records,columns=ERROR_COLUMNS).sort_values(['task','issue_id']).reset_index(drop=True),
            pd.DataFrame(source_checks))


def verify(decisions: pd.DataFrame, uses: pd.DataFrame, errors: pd.DataFrame,
           checks: pd.DataFrame, *, diagnostic:bool) -> None:
    assert decisions['task'].astype(int).tolist()==list(range(1,35)), 'Task 34 decision log is missing/reordered'
    assert uses.shape==(35,len(AI_COLUMNS)), f'Expected 35 AI-use entries, found {uses.shape}'
    assert uses.task.tolist()==list(range(1,36)), 'AI-use log must have Tasks 1–35 once each'
    assert 'PENDING_P2' in uses.loc[uses.task.eq(30),'verification_by_student_report'].iloc[0]
    assert 'PENDING_P2' in decisions.loc[decisions.task.eq(30),'validation_status'].iloc[0]
    assert ('TEAMS_UPLOAD_PENDING' in decisions.loc[decisions.task.eq(29),'validation_status'].iloc[0]
            or (diagnostic and 'MISSING_EVIDENCE' in decisions.loc[decisions.task.eq(29),'validation_status'].iloc[0])), (
                'Task 29 export must have Teams upload pending; partial decision log may lack original source')
    assert uses.loc[uses.task.eq(35),'evidence_kind'].iloc[0].startswith('CURRENT_')
    assert not errors.empty and errors.issue_id.is_unique, 'Error log must be nonempty and issue IDs unique'
    assert (errors['original_ai_claim_or_error'].map(available) &
            errors['correction_or_clarification'].map(available) &
            errors['source_evidence'].map(available)).all(), 'Every error must identify claim, correction and source'
    assert 18 in errors.task.to_list(), 'Task 18 dependency error must be recorded'
    assert (errors.source_record_json.eq('') | errors.source_record_json.str.startswith('{')).all()
    assert len(checks)==len(EXPECTED_ERROR_FILE_TASKS)
    if not diagnostic:
        assert checks['found'].all(), 'Missing original AI discrepancy evidence'
        assert not uses.evidence_kind.str.contains('MISSING_EVIDENCE|CSV_DIAGNOSTIC_ONLY').any()
        assert not checks['kind'].eq('CSV_DIAGNOSTIC_ONLY').any()


def make_review_checks(uses:pd.DataFrame, errors:pd.DataFrame, sources:pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {'check':'Tasks 1–35 each have a labelled AI-use row', 'passed':len(uses)==35 and uses.task.nunique()==35},
        {'check':'Tasks 1–12 explicitly retrospective', 'passed':uses.head(12).evidence_kind.str.startswith('RETROSPECTIVE_').all()},
        {'check':'Tasks 13–29 and 31–34 have original evidence', 'passed':(uses[uses.task.isin(SOURCE_AI_TASKS)].evidence_kind=='ORIGINAL_PER_TASK_AI_USE_CSV').all()},
        {'check':'All expected original discrepancy files present', 'passed':sources.found.all() and sources.kind.eq('ORIGINAL').all()},
        {'check':'Task 30 education integration still pending P2', 'passed':'PENDING_P2' in uses.loc[uses.task.eq(30),'verification_by_student_report'].iloc[0]},
        {'check':'All errors include an original and a correction', 'passed':errors.original_ai_claim_or_error.map(available).all() and errors.correction_or_clarification.map(available).all()},
        {'check':'Task 18 chat-reported import failure separately identified', 'passed':errors.issue_id.str.contains('T18-CHAT').any()},
        {'check':'No survey microdata used or written', 'passed':True},
    ])


def validate_task34_decision_log(root:Path, diagnostic:bool) -> pd.DataFrame:
    path=root/'task34_consolidated_decision_log.csv'
    if not path.exists():
        raise FileNotFoundError('Need completed Task 34 source: '+str(path))
    data=pd.read_csv(path,dtype=str,keep_default_na=False)
    if len(data)!=34:
        raise AssertionError('Task 34 decisions must have 34 rows')
    data['task']=pd.to_numeric(data['task'],errors='raise').astype(int)
    if not diagnostic:
        if data.validation_status.str.contains('MISSING_EVIDENCE|CSV_DIAGNOSTIC_ONLY').any():
            raise AssertionError('Task34 decision log is diagnostic or missing evidence')
    return data.sort_values('task').reset_index(drop=True)


def self_test() -> None:
    # Synthetic source files: NO real survey data used and no user files changed.
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        decisions=[]
        for task in range(1,35):
            status='PENDING_P2__NOT_COMPLETED' if task==30 else (
                'EXPORT_VERIFIED__TEAMS_UPLOAD_PENDING' if task==29 else 'SOURCE_VERIFIED')
            decisions.append({'task':task,'validation_status':status})
        pd.DataFrame(decisions).to_csv(root/'task34_consolidated_decision_log.csv',index=False)
        for task in SOURCE_AI_TASKS:
            # Mix schemas explicitly.
            row=({'task':task,'ai_use':f'Synthetic contribution {task}','verification':'synthetic check'}
                 if task in (19,20) else
                 {'task':task,'ai_contribution':f'Synthetic contribution {task}',
                  'human_verification':'pending student SAV' if task==14 else 'synthetic run'})
            pd.DataFrame([row]).to_csv(root/f'task{task:02d}_ai_use_log_entry.csv',index=False)
        for task in EXPECTED_ERROR_FILE_TASKS:
            pd.DataFrame([{'task':task,'error':f'Synthetic earlier issue {task}',
                           'how_detected':'Synthetic validation','correction':'Synthetic correction',
                           'status':'corrected'}]).to_csv(root/f'task{task:02d}_ai_discrepancy_entry.csv',index=False)
        decisions=validate_task34_decision_log(root,False)
        uses=make_ai_log(root,diagnostic=False)
        errors,checks=make_error_log(root,diagnostic=False)
        verify(decisions,uses,errors,checks,diagnostic=False)
        print('PASS: 35 unique AI-use records; 1–12 retrospective, 21 original task source files')
        assert uses.loc[uses.task.eq(14),'source_record_original_status'].iloc[0]=='pending student SAV'
        print('PASS: stale source verification wording preserved, not silently marked passed')
        assert uses.loc[uses.task.eq(30),'verification_by_student_report'].iloc[0].startswith('PENDING_P2')
        assert any('Task 18' in text for text in errors.original_ai_claim_or_error)
        print('PASS: real P2 merge and Teams upload remain pending; chat-reported import error separately labelled')
        assert len(errors)==len(EXPECTED_ERROR_FILE_TASKS)+len(CHAT_REPORTED_ERRORS)+len(DOC_VERIFIED_ERRORS)
        print('PASS: original discrepancy sources are attributed; synthetic errors normalized consistently')
        (root/'task19_ai_use_log_entry.csv').unlink()
        try:
            make_ai_log(root,diagnostic=False)
        except FileNotFoundError:
            print('PASS: missing original AI-use evidence makes final mode FAIL')
        else:
            raise AssertionError('Missing AI-use entry incorrectly accepted')
        partial=make_ai_log(root,diagnostic=True)
        assert partial.loc[partial.task.eq(19),'evidence_kind'].iloc[0].startswith('MISSING_EVIDENCE')
        print('PASS: local diagnostic reports missing evidence explicitly, never silently invents it')
        pd.DataFrame([{'task':19,'ai_use':'synthetic'}]).to_csv(root/'task19_ai_use_log_entry.csv',index=False)
        (root/'task23_ai_discrepancy_entry.csv').unlink()
        try:
            make_error_log(root,diagnostic=False)
        except FileNotFoundError:
            print('PASS: missing original discrepancy evidence makes final mode FAIL')
        else:
            raise AssertionError('Missing discrepancy source incorrectly accepted')
        pd.DataFrame([{'task':23,'error':'bad error without fix'}]).to_csv(
            root/'task23_ai_discrepancy_entry.csv',index=False)
        try:
            make_error_log(root,diagnostic=False)
        except AssertionError:
            print('PASS: AI error without a documented correction is rejected')
        else:
            raise AssertionError('Uncorrected error entry incorrectly accepted')
    print('TASK 35 SELF-TEST PASSED')


def main(*, diagnostic: bool) -> None:
    print('=== TASK 35: AI-USE & AI-DISCREPANCY LOG CONSOLIDATION ===')
    print('Mode:', 'PARTIAL LOCAL DIAGNOSTIC (NOT FINAL)' if diagnostic else 'LOCAL ORIGINAL TASK EVIDENCE (FINAL)')
    decisions=validate_task34_decision_log(OUT,diagnostic)
    uses=make_ai_log(OUT,diagnostic=diagnostic)
    errors,source_check=make_error_log(OUT,diagnostic=diagnostic)
    verify(decisions,uses,errors,source_check,diagnostic=diagnostic)
    review=make_review_checks(uses,errors,source_check)
    if not diagnostic and not review.passed.all():
        bad=review.loc[~review.passed,'check'].tolist()
        raise AssertionError('AI log review checks failed: '+str(bad))
    OUT.mkdir(parents=True,exist_ok=True)
    prefix='task35_partial_' if diagnostic else 'task35_'
    uses.to_csv(OUT/f'{prefix}consolidated_ai_use_log.csv',index=False)
    errors.to_csv(OUT/f'{prefix}consolidated_ai_discrepancy_log.csv',index=False)
    source_check.to_csv(OUT/f'{prefix}error_evidence_inventory.csv',index=False)
    review.to_csv(OUT/f'{prefix}review_checks.csv',index=False)
    report=(
      'Task 35: consolidated AI assistance and corrections.\n'
      f'AI-use entries: {len(uses)} (Tasks 1–35, one each).\n'
      f'Discrepancy entries: {len(errors)} (multiple per task are allowed).\n'
      f'Original per-task AI-use CSVs required: {len(SOURCE_AI_TASKS)}.\n'
      f'Original per-task discrepancy CSVs required: {len(EXPECTED_ERROR_FILE_TASKS)}.\n'
      'Tasks 1–12 are RETROSPECTIVE summaries from earlier conversation, not claimed as original task AI-use CSVs.\n'
      'The original error wording and original source status are preserved verbatim, including stale pending student-run labels.\n'
      'A student-reported successful rerun is noted separately from the source CSV status; source existence alone is not independent verification.\n'
      'Task 18 ModuleNotFoundError is a separately marked retrospective entry from the student conversation.\n'
      'No methodology caveat is automatically classified as an AI error.\n'
      'Task 29 private Teams upload PENDING until confirmed. Task 30 P2 real education join PENDING.\n'
      'No original data or existing task logs are modified. Audit outputs contain no respondent-level information.\n'
      'Student must read the consolidated logs, confirm the chronology and only submit claims supported by their terminal records.\n'
    )
    (OUT/f'{prefix}validation_note.txt').write_text(report,encoding='utf-8')
    print(f'AI-use coverage: {len(uses)}/35 tasks')
    print(f'Early retrospective descriptions: {len(EARLY_AI)} (labelled; review required)')
    print('Original per-task AI-use files:',int(uses.evidence_kind.eq('ORIGINAL_PER_TASK_AI_USE_CSV').sum()),
          '/',len(SOURCE_AI_TASKS))
    print('Original AI discrepancy sources:',int((source_check.kind=='ORIGINAL').sum()),
          '/',len(EXPECTED_ERROR_FILE_TASKS))
    print('Recorded AI discrepancies/corrections:',len(errors),
          '(including separately labelled chat-reported import issue)')
    print(f'Review checks: {int(review.passed.sum())}/{len(review)} passed')
    print('PENDING: Task 30 real P2 education merge; Task 29 restricted Teams upload')
    print('NOTE: earlier task log statuses may be stale; original wording is preserved for human review')
    print(f'Saved: {prefix}consolidated_ai_use_log.csv, {prefix}consolidated_ai_discrepancy_log.csv,')
    print(f'       {prefix}error_evidence_inventory.csv, {prefix}review_checks.csv, {prefix}validation_note.txt')
    if diagnostic:
        print('TASK 35 DIAGNOSTIC ONLY: missing real task evidence; NOT ready for submission.')
    else:
        print('TASK 35 PASSED: AI-use and error logs consolidated from local task evidence;')
        print('                P2 and Teams remain pending; no survey data changed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    switches=parser.add_mutually_exclusive_group()
    switches.add_argument('--self-test',action='store_true')
    switches.add_argument('--local-diagnostic',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main(diagnostic=args.local_diagnostic)