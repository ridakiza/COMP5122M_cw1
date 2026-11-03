"""COMP5122M P1 - Task 5: check the .sav and the CSV hold the same answers for the same people.

Run from the repository root:
    python scripts/validate_task05.py              # the real comparison
    python scripts/validate_task05.py --self-test  # proves the checker catches errors

Task 2 showed both files have the same size and country counts. This goes further:
every value of every column, person by person (matched on respondent id).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "clean"
SAV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset.sav"
CSV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv"

EXACT = ["id", "country", "weight", "ai_heard", "ai_cncexc", "age", "age_australia",
         "gender", "gender_australia", "internetfreq"]
PREFIXES = ("aireg_", "d_age_", "d_educ_")
WEIGHT_TOL = 1e-9   # weights are decimals; allow only rounding-level differences


def part_one_columns(cols) -> list[str]:
    return EXACT + sorted(c for c in cols if c.startswith(PREFIXES))


def compare(a: pd.DataFrame, b: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per column: how many values differ. Blank in both counts as equal."""
    rows = []
    for col in columns:
        x, y = a[col], b[col]
        xn = pd.to_numeric(x, errors="coerce")
        yn = pd.to_numeric(y, errors="coerce")
        numeric = xn.notna().sum() == x.notna().sum() and yn.notna().sum() == y.notna().sum()
        both_blank = x.isna() & y.isna()
        if not numeric:
            xd = pd.to_datetime(x, errors="coerce")
            yd = pd.to_datetime(y, errors="coerce")
            if xd.notna().sum() == x.notna().sum() and yd.notna().sum() == y.notna().sum():
                same = both_blank | (xd.dt.normalize() == yd.dt.normalize())
                rows.append({"variable": col, "type": "date",
                             "n_sav_present": int(x.notna().sum()), "n_csv_present": int(y.notna().sum()),
                             "n_mismatches": int((~same).sum())})
                continue
        if numeric:
            tol = WEIGHT_TOL if col == "weight" else 0
            same = both_blank | ((xn - yn).abs() <= tol)
        else:  # text/date columns: compare as trimmed text
            same = both_blank | (x.astype(str).str.strip() == y.astype(str).str.strip())
        rows.append({"variable": col, "type": "number" if numeric else "text",
                     "n_sav_present": int(x.notna().sum()), "n_csv_present": int(y.notna().sum()),
                     "n_mismatches": int((~same).sum())})
    return pd.DataFrame(rows)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    sav, _ = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    csv = pd.read_csv(CSV, low_memory=False, encoding="utf-8-sig")
    for df in (sav, csv):
        df.columns = df.columns.str.strip().str.lower()
    # The CSV stores blanks as a single space (" "), not as empty cells.
    # Turn whitespace-only cells into real blanks so they compare equal to the .sav's blanks.
    csv = csv.replace(r"^\s*$", np.nan, regex=True)
    assert sav.shape == csv.shape == (28_333, 483), f"Shapes: SAV {sav.shape}, CSV {csv.shape}"
    assert set(sav.columns) == set(csv.columns), "The two files have different column names"
    assert sav["id"].is_unique and csv["id"].is_unique
    # Match people by id, so a different row order cannot cause false mismatches.
    return sav.set_index("id", drop=False).sort_index(), csv.set_index("id", drop=False).sort_index()


def self_test() -> None:
    _, csv = load()
    cols = part_one_columns(csv.columns)
    copy = csv.copy(deep=True)
    assert compare(csv, copy, cols).n_mismatches.sum() == 0
    print("PASS: identical copies give zero mismatches")
    first = copy.index[0]
    copy.loc[first, "ai_heard"] = 3 if copy.loc[first, "ai_heard"] != 3 else 1
    copy.loc[first, "weight"] += 1e-6
    res = compare(csv, copy, cols)
    assert res.n_mismatches.sum() == 2 and set(res[res.n_mismatches > 0].variable) == {"ai_heard", "weight"}
    print("PASS: two deliberate errors (an answer and a weight) were both caught")
    print("TASK 5 SELF-TEST PASSED (tests the checker only)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    if parser.parse_args().self_test:
        return self_test()

    print("=== TASK 5: SAV VS CSV, PERSON BY PERSON ===")
    sav, csv = load()
    print(f"Both files: {len(sav):,} people x {sav.shape[1]} columns; same column names; matched on id")
    print(f"IDs identical in both files: {sav.index.equals(csv.index)}")

    part = compare(sav, csv, part_one_columns(sav.columns))
    print(f"\nPart I columns compared: {len(part)}")
    print(part.to_string(index=False))
    every = compare(sav, csv, list(sav.columns))
    bad = every[every.n_mismatches > 0]

    OUT.mkdir(parents=True, exist_ok=True)
    part.to_csv(OUT / "task05_part1_comparison.csv", index=False)
    every.to_csv(OUT / "task05_all_columns_comparison.csv", index=False)

    print(f"\nPart I mismatching values: {int(part.n_mismatches.sum())}")
    print(f"All 483 columns, mismatching values: {int(every.n_mismatches.sum())}")
    if len(bad):
        print("Columns with differences:\n" + bad.to_string(index=False))
    assert part.n_mismatches.sum() == 0, "TASK 5 FAILED: Part I values differ between SAV and CSV"
    print("\nSaved: task05_part1_comparison.csv, task05_all_columns_comparison.csv")
    print("\nTASK 5 PASSED: the .sav and CSV give identical answers for every person in all 64 Part I columns.")


if __name__ == "__main__":
    main()