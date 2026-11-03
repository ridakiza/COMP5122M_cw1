"""COMP5122M Part I — Task 6: detect duplicate IDs and identical original rows.

From the repository root:
    python scripts/validate_task06.py --self-test
    python scripts/validate_task06.py

The original SAV is authoritative; CSV is an independent export cross-check.
No rows are deleted or altered. Missing values are compared by pandas as equal
for duplicate detection (including two blank cells in the same column).

Optional diagnostic mode (does not complete Task 6 on its own):
    python scripts/validate_task06.py --csv-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "clean"
SAV_PATH = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset.sav"
CSV_PATH = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv"
EXPECTED_ROWS = 28_333
EXPECTED_COLUMNS = 483


def duplicate_counts(source_df: pd.DataFrame, source: str) -> tuple[dict, pd.DataFrame]:
    """Audit untouched respondent records; return totals and any flagged IDs."""
    # Normalise *headers only* in a separate DataFrame; never clean survey values.
    df = source_df.rename(columns=lambda col: str(col).strip().lower())
    if not df.columns.is_unique:
        raise AssertionError(f"{source}: duplicate column names after normalisation")
    if not {"id", "country"}.issubset(df.columns):
        raise AssertionError(f"{source}: 'id' or 'country' is missing")

    # Full-row checks include all 483 source fields. Duplicated treats NaN=NaN.
    duplicated_id = df["id"].duplicated(keep=False)
    duplicated_full = df.duplicated(keep=False)
    # Stronger diagnostic: two different IDs may have identical other 482 values.
    duplicated_without_id = df.drop(columns=["id"]).duplicated(keep=False)
    # Strictest check: same answers to every question, ignoring fields that differ
    # between interviews even for a copied record (ID, weight, interview dates).
    admin = [c for c in ("id", "weight", "qdate_s", "qdate_e", "intdate_aus") if c in df.columns]
    duplicated_answers = df.drop(columns=admin).duplicated(keep=False)

    counts = {
        "source": source,
        "rows": len(df),
        "columns": df.shape[1],
        "missing_ids": int(df["id"].isna().sum()),
        "duplicate_id_rows": int(duplicated_id.sum()),
        "identical_full_rows": int(duplicated_full.sum()),
        "identical_rows_excluding_id": int(duplicated_without_id.sum()),
        "identical_answers_only": int(duplicated_answers.sum()),
    }
    counts["passed"] = (
        counts["rows"] == EXPECTED_ROWS
        and counts["columns"] == EXPECTED_COLUMNS
        and all(counts[key] == 0 for key in (
            "missing_ids", "duplicate_id_rows", "identical_full_rows",
            "identical_rows_excluding_id", "identical_answers_only",
        ))
    )

    any_flag = duplicated_id | duplicated_full | duplicated_without_id | duplicated_answers
    flagged = df.loc[any_flag, ["id", "country"]].copy()
    if not flagged.empty:
        flagged.insert(0, "source", source)
        flagged["duplicate_id"] = duplicated_id.loc[any_flag].to_numpy()
        flagged["identical_full_row"] = duplicated_full.loc[any_flag].to_numpy()
        flagged["identical_excluding_id"] = duplicated_without_id.loc[any_flag].to_numpy()
        flagged["identical_answers_only"] = duplicated_answers.loc[any_flag].to_numpy()
    return counts, flagged


def self_test() -> None:
    """Deliberately plant an ID collision, a full duplicate and an ID-only difference."""
    sample = pd.DataFrame({
        "id": [1, 2, 2, 3, 4],
        "country": [14, 14, 14, 13, 14],
        "weight": [0.5, 1.0, 1.0, 1.0, 0.5],
        "ai_heard": [1, 2, 2, 3, 1],
    })
    result, flagged = duplicate_counts(sample, "synthetic_test")
    assert result["duplicate_id_rows"] == 2, result
    assert result["identical_full_rows"] == 2, result
    assert result["identical_rows_excluding_id"] == 4, result
    assert len(flagged) == 4, result

    unique = sample.iloc[[0, 1, 3]].copy()
    unique_result, _ = duplicate_counts(unique, "unique_synthetic_test")
    assert unique_result["duplicate_id_rows"] == 0
    assert unique_result["identical_full_rows"] == 0
    assert unique_result["identical_rows_excluding_id"] == 0
    print("PASS: synthetic duplicate IDs detected")
    print("PASS: synthetic full-row duplicates detected")
    print("PASS: identical survey answers under different IDs detected")
    print("PASS: unique synthetic rows have no duplicates")
    print("TASK 6 SELF-TEST PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--csv-only", action="store_true", help="diagnostic only, not final SAV verification")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Missing original CSV: {CSV_PATH}")
    if not args.csv_only and not SAV_PATH.is_file():
        raise FileNotFoundError(f"Missing original SAV: {SAV_PATH}")

    sources: list[tuple[str, pd.DataFrame]] = []
    if not args.csv_only:
        import pyreadstat
        sav, _metadata = pyreadstat.read_sav(
            str(SAV_PATH), apply_value_formats=False, user_missing=True
        )
        sources.append(("SAV", sav))
    csv_df = pd.read_csv(CSV_PATH, low_memory=False, encoding="utf-8-sig")
    sources.append(("CSV", csv_df))

    results: list[dict] = []
    flagged_frames: list[pd.DataFrame] = []
    print("=== TASK 6: DUPLICATE RESPONDENTS AND ROWS ===")
    for source, df in sources:
        counts, flagged = duplicate_counts(df, source)
        results.append(counts)
        if not flagged.empty:
            flagged_frames.append(flagged)
        print(f"\n{source}: {counts['rows']:,} rows x {counts['columns']} columns")
        print(f"  IDs missing:                       {counts['missing_ids']:,}")
        print(f"  Rows with a repeated ID:           {counts['duplicate_id_rows']:,}")
        print(f"  Rows identical across 483 fields:  {counts['identical_full_rows']:,}")
        print(f"  Rows identical excluding ID:       {counts['identical_rows_excluding_id']:,}")
        print(f"  Same answers (ignoring ID, weight, interview dates): {counts['identical_answers_only']:,}")
        print(f"  {'PASS' if counts['passed'] else 'FAIL'}: {source} duplicate checks")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT / "task06_duplicate_audit.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    if flagged_frames:
        details_path = OUTPUT / "task06_duplicate_examples.csv"
        pd.concat(flagged_frames, ignore_index=True).to_csv(details_path, index=False)
        print(f"Review flagged records (private): {details_path}")

    if not all(row["passed"] for row in results):
        raise AssertionError("TASK 6 FAILED: investigate flagged records; do NOT delete them automatically.")
    if args.csv_only:
        print("\nTASK 6 CSV DIAGNOSTIC PASSED — SAV validation still required.")
    else:
        print("\nTASK 6 PASSED: SAV + CSV confirm no duplicate IDs or identical rows.")


if __name__ == "__main__":
    main()