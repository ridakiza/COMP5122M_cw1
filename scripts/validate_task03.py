"""COMP5122M P1 — Task 3: select Part I variables without changing the raw dataset.

Run from the repository root:
    python scripts/validate_task03.py

Independent CSV diagnostic (does NOT replace SAV verification):
    python scripts/validate_task03.py --csv-only

No respondent-level output is written; only a column manifest is exported.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "clean"
SAV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset.sav"
CSV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv"

# Columns explicitly required in Task 3.
EXACT = [
    "id", "country", "weight", "ai_heard", "ai_cncexc",
    "age", "age_australia", "gender", "gender_australia", "internetfreq",
]

# Independent reference list checked against the uploaded Pew CSV's headers.
# This is intentionally explicit: it catches omitted/wrong columns even if
# a dynamic prefix-based selection happens to have the expected length.
EXPECTED_FAMILIES = {
    "aireg_": [
        "aireg_us", "aireg_china", "aireg_eu", "aireg_country",
    ],
    "d_age_": [
        "d_age_argentina", "d_age_brazil", "d_age_canada", "d_age_france",
        "d_age_germany", "d_age_greece", "d_age_hungary", "d_age_india",
        "d_age_indonesia", "d_age_israel", "d_age_italy", "d_age_japan",
        "d_age_kenya", "d_age_mexico", "d_age_netherlands", "d_age_nigeria",
        "d_age_poland", "d_age_safrica", "d_age_skorea", "d_age_spain",
        "d_age_sweden", "d_age_turkey", "d_age_uk",
    ],
    "d_educ_": [
        "d_educ_argentina", "d_educ_australia", "d_educ_brazil",
        "d_educ_canada", "d_educ_france", "d_educ_germany",
        "d_educ_germany_vocational", "d_educ_greece", "d_educ_hungary",
        "d_educ_india", "d_educ_indonesia", "d_educ_israel",
        "d_educ_israel_years", "d_educ_israel_vocational", "d_educ_italy",
        "d_educ_japan", "d_educ_kenya", "d_educ_mexico",
        "d_educ_netherlands", "d_educ_nigeria", "d_educ_poland",
        "d_educ_safrica", "d_educ_skorea", "d_educ_spain",
        "d_educ_sweden", "d_educ_turkey", "d_educ_uk",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-only", action="store_true", help="Diagnostics only; SAV remains authoritative")
    args = parser.parse_args()

    if args.csv_only:
        raw_df = pd.read_csv(CSV, low_memory=False, encoding="utf-8-sig")
        source = "CSV_DIAGNOSTIC"
    else:
        try:
            import pyreadstat
        except ImportError as exc:
            raise RuntimeError("Install dependencies: python -m pip install -r requirements.txt") from exc
        # Keep numeric answers, user-defined missing values and original headers.
        raw_df, _meta = pyreadstat.read_sav(
            str(SAV), apply_value_formats=False, user_missing=True
        )
        source = "SAV"

    # Save a snapshot for an exact after-selection comparison.
    original_snapshot = raw_df.copy(deep=True)
    original_column_names = list(raw_df.columns)
    assert raw_df.shape == (28_333, 483), f"Unexpected raw shape: {raw_df.shape}"

    # Canonical lowercase names only in a *lookup*; don't edit raw_df.
    normalized = [str(col).strip().lower() for col in raw_df.columns]
    assert len(set(normalized)) == len(normalized), "Column names collide after case normalization"
    source_name = dict(zip(normalized, raw_df.columns))

    # Select the fields dynamically based on the exact and family specifications.
    prefixes = tuple(EXPECTED_FAMILIES)
    selected = EXACT + sorted(col for col in normalized if col.startswith(prefixes))

    # Double verification: independent explicit reference lists for every family.
    expected = set(EXACT)
    expected.update(c for family in EXPECTED_FAMILIES.values() for c in family)
    missing = sorted(expected - set(selected))
    unexpected = sorted(set(selected) - expected)
    assert not missing and not unexpected, (
        f"Selection differs from checked dataset headers. Missing: {missing}; unexpected: {unexpected}"
    )
    assert all(name in source_name for name in selected), "A required field is absent from source"
    assert len(selected) == len(set(selected)) == 64, "Expected 64 distinct Part I columns"

    # Deep copy selected values; only the selected DataFrame gets standardised names.
    selected_df = raw_df.loc[:, [source_name[name] for name in selected]].copy(deep=True)
    selected_df.columns = selected

    # Confirm no rows are removed, data altered, or indices reordered.
    assert selected_df.shape == (28_333, 64)
    assert selected_df.index.equals(raw_df.index), "Rows or row ordering changed"
    pd.testing.assert_frame_equal(
        selected_df,
        raw_df.loc[:, [source_name[name] for name in selected]].set_axis(selected, axis=1),
    )
    # Confirm the original DataFrame — including headers and all 483 values — is intact.
    pd.testing.assert_frame_equal(raw_df, original_snapshot)
    assert list(raw_df.columns) == original_column_names
    assert selected_df is not raw_df

    print("=== TASK 3: SELECT PART I VARIABLES ===")
    print(f"Source: {source}")
    print(f"Raw dataset: {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns")
    print(f"Selected dataset: {selected_df.shape[0]:,} rows x {selected_df.shape[1]} columns")
    print(f"Exact fields: {len(EXACT)}")
    for prefix, reference_fields in EXPECTED_FAMILIES.items():
        found = [name for name in selected if name.startswith(prefix)]
        print(f"{prefix}: {len(found)} columns (reference: {len(reference_fields)})")
        assert set(found) == set(reference_fields), f"Unexpected {prefix} fields"
    print("PASS: all 64 required columns were selected (verified against independent reference)")
    print("PASS: all 28,333 rows and their original order were preserved")
    print("PASS: selected values match source exactly")
    print("PASS: raw_df is unchanged (all 483 columns, headers, values, and index)")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame({
        "variable": selected,
        "source_column": [source_name[name] for name in selected],
        "family": [
            next((prefix for prefix in EXPECTED_FAMILIES if name.startswith(prefix)), "exact")
            for name in selected
        ],
        "dtype": [str(selected_df[name].dtype) for name in selected],
        "n_non_null_before_cleaning": [int(selected_df[name].notna().sum()) for name in selected],
        "validation_source": source,
    })
    output = OUT / "task03_column_manifest.csv"
    manifest.to_csv(output, index=False)
    print(f"Saved metadata-only manifest: {output}")
    print("TASK 3 PASSED" if source == "SAV" else "CSV DIAGNOSTIC PASSED; run without --csv-only for Task 3 final verification")


if __name__ == "__main__":
    main()