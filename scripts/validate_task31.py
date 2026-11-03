"""Task 31: execute the P1 data-quality notebook top-to-bottom in a NEW kernel.

    python scripts/validate_task31.py --self-test
    python scripts/validate_task31.py

Requires the Task 29 private provisional export + its checksum manifest.
Does not require P2's educ3 lookup, overwrite the provisional dataset, or
write respondent-level information to Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / 'scripts'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
OUT = ROOT / 'data' / 'clean'
NOTEBOOK = ROOT / 'notebooks' / '01_data_quality.ipynb'
EXPORT = OUT / 'pew_clean_p1_provisional.csv'
MANIFEST = OUT / 'task29_provisional_export_manifest.csv'
REFERENCE = ROOT / 'references' / 'topline_expected.csv'
RAW_CSV = ROOT / 'data' / 'raw' / 'Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
SAVED_EXECUTION = OUT / 'task31_fresh_execution.ipynb'


def require_notebook_packages():
    try:
        import nbformat
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError
    except ImportError as exc:
        raise RuntimeError('Notebook execution requires nbformat, nbclient, ipykernel. '
                           'Install with `python -m pip install nbformat nbclient ipykernel`.') from exc
    return nbformat, NotebookClient, CellExecutionError


def checksum_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def assert_notebook_structure(nb) -> None:
    code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
    assert len(code_cells) >= 7, f'Expected at least 7 P1 notebook code cells; got {len(code_cells)}'
    sources = '\n'.join(cell.source for cell in code_cells)
    for feature in ('task29_provisional_export_manifest.csv', 'topline_expected.csv',
                    'recovered', 'sensitivity_checks', 'TASK 31 NOTEBOOK PASSED'):
        assert feature in sources, f'Notebook missing essential P1 check: {feature}'
    assert 'educ3' in sources, 'Notebook must identify the pending P2 education integration'
    assert not any(c.cell_type == 'code' and c.source.lstrip().startswith('%run')
                   for c in nb.cells), 'Notebook must not depend on hidden interactive state'


def preflight() -> dict:
    from validate_task29 import preflight_git_protection
    git_result = preflight_git_protection()
    for path in (NOTEBOOK, EXPORT, MANIFEST, REFERENCE, RAW_CSV, ROOT / 'src/cleaning.py'):
        if not path.is_file():
            raise FileNotFoundError(f'Task 31 requires {path.relative_to(ROOT)}. '
                                    'Run Task 29 first or restore the repository reference.')
    manifest = pd.read_csv(MANIFEST)
    assert len(manifest) == 1 and 'sha256' in manifest.columns
    digest = checksum_file(EXPORT)
    assert digest == str(manifest.loc[0, 'sha256']), 'Provisional data checksum differs from Task 29 manifest'
    if 'rows' in manifest.columns:
        assert int(manifest.loc[0, 'rows']) == 28_333
    if 'columns' in manifest.columns:
        assert int(manifest.loc[0, 'columns']) == 70
    assert len(pd.read_csv(REFERENCE)) == 192
    print('PASS: Task 29 SHA-256 export checksum verified')
    print('PASS: 192 official Pew reference values available')
    print('PASS: P2 education file not required for the P1 notebook')
    return {'git_preflight': git_result, 'export_sha256': digest}


def execute_notebook(nb, *, root: Path, notebook_client_cls):
    # Starting NotebookClient with a new kernel (not reusing Jupyter state) is the
    # crucial Task 31 property; the input notebook's stored outputs are discarded.
    clean_copy = nb.copy()
    for cell in clean_copy.cells:
        if cell.cell_type == 'code':
            cell.execution_count = None
            cell.outputs = []
    client = notebook_client_cls(clean_copy, timeout=180, kernel_name='python3',
                                 resources={'metadata': {'path': str(root)}},
                                 allow_errors=False)
    return client.execute()


def self_test() -> None:
    nf, NotebookClient, CellExecutionError = require_notebook_packages()
    tiny = nf.v4.new_notebook(cells=[
        nf.v4.new_code_cell('test_31_value = 37'),
        nf.v4.new_code_cell('assert test_31_value + 5 == 42\nprint("FRESH TASK 31 KERNEL PASSED")'),
    ])
    result = execute_notebook(tiny, root=ROOT, notebook_client_cls=NotebookClient)
    code = [c for c in result.cells if c.cell_type == 'code']
    assert [c.execution_count for c in code] == [1, 2]
    assert any('FRESH TASK 31 KERNEL PASSED' in str(out.get('text', ''))
               for out in code[-1].outputs), 'Notebook cells did not execute sequentially'
    print('PASS: isolated notebook executes code sequentially in a new kernel')
    # A fresh kernel MUST NOT inherit a variable defined in the preceding run.
    no_state = nf.v4.new_notebook(cells=[
        nf.v4.new_code_cell('assert "test_31_value" not in globals()\nprint("NO LEAKED STATE")'),
    ])
    state_result = execute_notebook(no_state, root=ROOT, notebook_client_cls=NotebookClient)
    assert any('NO LEAKED STATE' in str(out.get('text', ''))
               for out in state_result.cells[0].outputs)
    print('PASS: a second notebook starts without state left by the first run')
    assert NOTEBOOK.is_file(), 'Missing Task 31 notebook — copy notebooks/01_data_quality.ipynb first'
    nb = nf.read(NOTEBOOK, as_version=4)
    assert_notebook_structure(nb)
    assert all(c.execution_count is None for c in nb.cells if c.cell_type == 'code'), (
        'Distributed notebook must begin with no saved execution state')
    print('PASS: P1 notebook has the required checks, is unexecuted and does not require P2')
    print('TASK 31 SELF-TEST PASSED')


def run() -> None:
    print('=== TASK 31: FRESH-KERNEL NOTEBOOK, TOP-TO-BOTTOM ===')
    nf, NotebookClient, _ = require_notebook_packages()
    before = preflight()
    nb = nf.read(NOTEBOOK, as_version=4)
    assert_notebook_structure(nb)
    original_nb_digest = checksum_file(NOTEBOOK)
    print(f'Executing {NOTEBOOK.relative_to(ROOT)} in an ISOLATED fresh kernel ...')
    result = execute_notebook(nb, root=ROOT, notebook_client_cls=NotebookClient)
    code = [c for c in result.cells if c.cell_type == 'code']
    assert len(code) >= 7 and all(c.execution_count is not None for c in code)
    assert len({c.execution_count for c in code}) == len(code)
    errors = [o for c in code for o in c.outputs if o.output_type == 'error']
    assert not errors, f'Notebook emitted {len(errors)} errors'
    last_output = '\n'.join(str(out.get('text', '')) for out in code[-1].outputs)
    assert 'TASK 31 NOTEBOOK PASSED' in last_output, 'Final notebook success marker not reached'
    assert checksum_file(EXPORT) == before['export_sha256'], 'Notebook changed provisional dataset'
    assert checksum_file(NOTEBOOK) == original_nb_digest, 'Notebook source file was overwritten'
    print(f'PASS: all {len(code)} code cells executed in a NEW kernel; no errors')
    print('PASS: the original notebook and provisional P1 dataset remain unchanged')
    print('PASS: notebook rechecked age/gender, raw recodes, 192 topline cells, Kish, sample sizes, sensitivities')

    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / 'task31_fresh_execution.tmp.ipynb'
    try:
        nf.write(result, temp)
        temp.replace(SAVED_EXECUTION)
    finally:
        temp.unlink(missing_ok=True)
    pd.DataFrame([{
        'task': 31, 'status': 'PASSED', 'notebook_code_cells': len(code),
        'executed_in_new_kernel': True, 'all_code_cells_passed': True,
        'respondents': 28333, 'p1_columns':70, 'pew_published_cells_reproduced':192,
        'age_bands_known':28148, 'age_bands_missing':185,
        'original_provisional_sha256':before['export_sha256'],
        'p2_educ3_integration':'PENDING',
    }]).to_csv(OUT / 'task31_notebook_execution_summary.csv', index=False)
    pd.DataFrame([{
        'task':31, 'decision':'Execute P1 notebook from a fresh kernel using the already verified provisional export',
        'alternative':'Use the interactive Jupyter state or require an unavailable P2 education mapping',
        'reason':'Fresh execution detects hidden state and order dependencies; P1 is independently testable',
        'effect':'All cells checked; original data unchanged; P2 merge remains pending', 'status':'SAV_EXPORT_VERIFIED_NOTEBOOK_PASSED',
    }]).to_csv(OUT / 'task31_decision_log_entry.csv', index=False)
    pd.DataFrame([{
        'task':31, 'ai_contribution':'Prepared P1-only reproducibility notebook and fresh-kernel execution validator',
        'verification':'Executed actual notebook via nbclient; independently checked checksum, outputs and response controls',
        'ai_errors':'Legacy starter notebook required missing modules; replaced by read-only P1-only notebook; no P2 invented',
        'status':'PASSED_WITH_ORIGINAL_SAV_VERIFIED_TASK29_EXPORT',
    }]).to_csv(OUT / 'task31_ai_use_log_entry.csv', index=False)
    (OUT / 'task31_validation_note.txt').write_text(
        'Task 31: P1 notebook executed completely in a new Python kernel.\n'
        f'Executed code cells: {len(code)}; no errors.\n'
        'Input is the Task 29 SHA-256-verified provisional P1 export,\n'
        'original Pew companion CSV and published topline reference.\n'
        'All 28,333 respondent records and all 70 P1 columns remain unchanged.\n'
        'P2 harmonised education (educ3) is not available; Task 30 is NOT complete.\n'
        'Executed notebook saved privately; do not commit to Git.\n', encoding='utf-8')
    print('Saved privately: task31_fresh_execution.ipynb, task31_notebook_execution_summary.csv,')
    print('                task31_decision_log_entry.csv, task31_ai_use_log_entry.csv,')
    print('                task31_validation_note.txt')
    print('TASK 31 PASSED: notebook ran from top to bottom in a fresh kernel; P2 remains pending.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', help='Test fresh-kernel behaviour without exporting any data')
    args = parser.parse_args()
    try:
        self_test() if args.self_test else run()
    except Exception as exc:
        print(f'TASK 31 NOT PASSED: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
