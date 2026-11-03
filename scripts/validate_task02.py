"""COMP5122M P1 - Task 2: check the original Pew files are complete and consistent.

Run from the repository root:
    python scripts/validate_task02.py

What it checks
1. The .sav has the size the assessment brief states: 28,333 respondents, 24 countries.
2. It has 483 columns, every respondent has a country, and IDs are unique.
3. The CSV (a separate export of the same survey) gives exactly the same
   dimensions and the same number of people in every country.
Country sizes are not typed in: they are counted from the .sav and confirmed by the CSV.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "clean"
SAV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset.sav"
CSV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv"

# Stated in the COMP5122M Part I brief ("28,333 in total", "24 countries").
BRIEF_RESPONDENTS = 28_333
BRIEF_COUNTRIES = 24
EXPECTED_COLUMNS = 483


def summarise(df: pd.DataFrame, names: dict, source: str) -> pd.DataFrame:
    """Basic checks on one file; returns people per country."""
    df.columns = df.columns.str.strip().str.lower()
    country = pd.to_numeric(df["country"], errors="raise")

    print(f"\n{source}: rows={len(df):,}; columns={df.shape[1]}; countries={country.nunique()}")
    assert len(df) == BRIEF_RESPONDENTS, f"{source}: {len(df)} rows, brief says {BRIEF_RESPONDENTS}"
    assert country.nunique() == BRIEF_COUNTRIES, f"{source}: {country.nunique()} countries, brief says {BRIEF_COUNTRIES}"
    assert df.shape[1] == EXPECTED_COLUMNS, f"{source}: {df.shape[1]} columns"
    assert country.notna().all(), f"{source}: some respondents have no country"
    assert not df["id"].duplicated().any(), f"{source}: duplicate respondent IDs"

    counts = country.astype(int).value_counts().sort_index()
    table = pd.DataFrame({
        "country_code": counts.index,
        "country": [names[float(c)] for c in counts.index],
        "n": counts.values,
    })
    assert table["n"].sum() == BRIEF_RESPONDENTS, f"{source}: country counts do not add up"
    print(f"PASS: {source} matches the brief (28,333 people, 24 countries), 483 columns, unique IDs.")
    return table


def main() -> None:
    print("=== TASK 2: DATASET LOADING AND SAMPLE-SIZE CHECKS ===")
    for path in (SAV, CSV):
        if not path.is_file():
            raise FileNotFoundError(f"Missing: {path}\nPut the original Pew file in data/raw/ with its original name.")

    sav, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    names = meta.variable_value_labels["country"]   # country names from the .sav labels
    sav_table = summarise(sav, names, "SAV")

    csv = pd.read_csv(CSV, low_memory=False, encoding="utf-8-sig")
    csv_table = summarise(csv, names, "CSV")

    both = sav_table.merge(csv_table, on=["country_code", "country"], suffixes=("_sav", "_csv"))
    both["match"] = both["n_sav"] == both["n_csv"]
    print("\nPeople per country (SAV vs CSV):")
    print(both.to_string(index=False))
    assert len(both) == BRIEF_COUNTRIES and both["match"].all(), "SAV and CSV country sizes differ"

    print(f"\nSmallest: {both.loc[both.n_sav.idxmin(), 'country']} ({both.n_sav.min():,}); "
          f"largest: {both.loc[both.n_sav.idxmax(), 'country']} ({both.n_sav.max():,})")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / "task02_country_counts.csv"
    both.to_csv(out, index=False)
    print("Saved:", out)
    print("\nTASK 2 PASSED: totals match the brief, and SAV and CSV agree on all 24 country sizes.")


if __name__ == "__main__":
    main()