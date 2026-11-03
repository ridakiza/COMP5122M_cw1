"""COMP5122M P1 — Task 9: verify survey weights in the original Pew files.

Run from the repository root:
    python scripts/validate_task09.py --self-test
    python scripts/validate_task09.py

Task 9 requirements: no missing weights; weights are strictly positive;
within each of the 24 countries, the sum of weights equals that country's
sample size; overall weight range is approximately 0.06 to 5.89.

The SAV is the authoritative source. The CSV is an independent export checked
respondent by respondent using ID. This script reads original files and writes
only summary/validation artifacts in data/clean; it never edits respondent data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "clean"
SAV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset.sav"
CSV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv"

EXPECTED_ROWS = 28_333
EXPECTED_COUNTRIES = 24
FIELDS = ["id", "country", "weight"]
SUM_TOLERANCE = 1e-6  # allowable round-off for a country's weight sum
VALUE_TOLERANCE = 1e-10  # SAV-vs-CSV precision for individual weights


def normalized(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Make a numeric working copy; do not alter the input DataFrame."""
    df = frame.copy(deep=True)
    df.columns = [str(c).strip().lower() for c in df.columns]
    assert df.columns.is_unique, f"{source}: duplicate column names"
    assert set(FIELDS).issubset(df.columns), f"{source}: missing {set(FIELDS) - set(df.columns)}"
    df = df[FIELDS].copy()
    for col in FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        assert df[col].notna().all(), f"{source}: {col} contains missing or non-numeric values"
        assert np.isfinite(df[col].to_numpy(dtype=float)).all(), f"{source}: {col} contains nonfinite values"
    assert (df["weight"] > 0).all(), f"{source}: weight must be strictly positive"
    for col in ["id", "country"]:
        assert (df[col] == np.floor(df[col])).all(), f"{source}: {col} has non-integer entries"
        df[col] = df[col].astype("int64")
    assert df["id"].is_unique, f"{source}: respondent IDs are not unique"
    return df


def audit_weights(frame: pd.DataFrame, source: str,
                  expected_rows: int | None = EXPECTED_ROWS,
                  expected_countries: int | None = EXPECTED_COUNTRIES,
                  check_published_range: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate each weight and each country's normalisation."""
    df = normalized(frame, source)
    if expected_rows is not None:
        assert len(df) == expected_rows, f"{source}: expected {expected_rows} respondents, got {len(df)}"
    if expected_countries is not None:
        assert df["country"].nunique() == expected_countries, f"{source}: unexpected number of countries"

    by_country = (
        df.groupby("country", sort=True)
        .agg(n_respondents=("id", "size"),
             sum_weight=("weight", "sum"),
             mean_weight=("weight", "mean"),
             min_weight=("weight", "min"),
             max_weight=("weight", "max"))
        .reset_index().rename(columns={"country": "country_code"})
    )
    by_country["weight_minus_n"] = by_country["sum_weight"] - by_country["n_respondents"]
    by_country["normalised"] = by_country["weight_minus_n"].abs() <= SUM_TOLERANCE

    global_min = float(df["weight"].min())
    global_max = float(df["weight"].max())
    total_weight = float(df["weight"].sum())
    summary = pd.DataFrame([{
        "source": source,
        "n_respondents": len(df),
        "n_countries": df["country"].nunique(),
        "n_missing_weights": int(df["weight"].isna().sum()),
        "n_nonpositive_weights": int((df["weight"] <= 0).sum()),
        "sum_weight": total_weight,
        "min_weight": global_min,
        "max_weight": global_max,
        "n_countries_normalised": int(by_country["normalised"].sum()),
        "largest_abs_country_sum_error": float(by_country["weight_minus_n"].abs().max()),
    }])

    assert by_country["normalised"].all(), (
        f"{source}: weight sums differ from sample sizes: "
        f"{by_country.loc[~by_country.normalised, ['country_code', 'weight_minus_n']].to_dict('records')}"
    )
    assert abs(total_weight - len(df)) <= SUM_TOLERANCE, f"{source}: global weight sum differs from respondent count"
    if check_published_range:
        # The brief gives values rounded to TWO decimal places, not exact endpoints.
        assert round(global_min, 2) == 0.06, f"{source}: minimum {global_min} is not approximately 0.06"
        assert round(global_max, 2) == 5.89, f"{source}: maximum {global_max} is not approximately 5.89"
    return by_country, summary


def compare_sources(sav: pd.DataFrame, csv: pd.DataFrame) -> tuple[int, int, float]:
    """Match on ID (not row position), checking countries and individual weights."""
    a = normalized(sav, "SAV").set_index("id").sort_index()
    b = normalized(csv, "CSV").set_index("id").sort_index()
    assert a.index.equals(b.index), "SAV and CSV respondent IDs differ"
    country_mismatches = int((a["country"] != b["country"]).sum())
    differences = (a["weight"] - b["weight"]).abs()
    matching_weight = np.isclose(a["weight"].to_numpy(dtype=float),
                                 b["weight"].to_numpy(dtype=float),
                                 atol=VALUE_TOLERANCE, rtol=0)
    weight_mismatches = int((~matching_weight).sum())
    max_diff = float(differences.max())
    assert country_mismatches == 0, f"SAV and CSV: {country_mismatches} country mismatches"
    assert weight_mismatches == 0, f"SAV and CSV: {weight_mismatches} weight mismatches"
    return country_mismatches, weight_mismatches, max_diff


def must_fail(func, description: str) -> None:
    try:
        func()
    except (AssertionError, ValueError):
        print(f"PASS: {description}")
    else:
        raise AssertionError(f"Self-test failed to catch: {description}")


def self_test() -> None:
    toy = pd.DataFrame({
        "id": [1, 2, 3, 4], "country": [1, 1, 2, 2],
        "weight": [0.5, 1.5, 0.75, 1.25]
    })
    by_country, _ = audit_weights(toy, "TOY", expected_rows=4,
                                   expected_countries=2,
                                   check_published_range=False)
    assert by_country["normalised"].all()
    print("PASS: correctly normalised weights accepted")
    damaged = toy.copy()
    damaged.loc[0, "weight"] = np.nan
    must_fail(lambda: audit_weights(damaged, "TOY", 4, 2, False), "missing weights rejected")
    damaged = toy.copy()
    damaged.loc[0, "weight"] = -0.5
    must_fail(lambda: audit_weights(damaged, "TOY", 4, 2, False), "negative weights rejected")
    damaged = toy.copy()
    damaged.loc[0, "weight"] = 0.7
    must_fail(lambda: audit_weights(damaged, "TOY", 4, 2, False), "unnormalised country weights rejected")
    damaged = toy.copy()
    damaged.loc[0, "weight"] = 0.6
    damaged.loc[1, "weight"] = 1.4
    must_fail(lambda: compare_sources(toy, damaged), "individual SAV/CSV differences rejected even when sums match")
    print("TASK 9 SELF-TEST PASSED")


def main(csv_only: bool = False) -> None:
    print("=== TASK 9: SURVEY WEIGHT VALIDATION (ORIGINAL FILES) ===")
    if not CSV.is_file():
        raise FileNotFoundError(f"Missing CSV file: {CSV}")
    csv = pd.read_csv(CSV, usecols=lambda x: x.strip().lower() in FIELDS,
                      low_memory=False, encoding="utf-8-sig")
    csv_country, csv_total = audit_weights(csv, "CSV")

    if csv_only:
        print("CSV-ONLY REFERENCE CHECK — not final Task 9 verification")
        print(csv_total.to_string(index=False))
        return

    if not SAV.is_file():
        raise FileNotFoundError(f"Missing SAV file: {SAV}")
    try:
        import pyreadstat
    except ImportError as exc:
        raise ImportError("Install pyreadstat using: python -m pip install pyreadstat") from exc
    sav, meta = pyreadstat.read_sav(str(SAV), usecols=FIELDS,
                                   apply_value_formats=False, user_missing=True)
    sav_country, sav_total = audit_weights(sav, "SAV")
    n_country_diff, n_weight_diff, max_weight_diff = compare_sources(sav, csv)

    country_labels = meta.variable_value_labels.get("country", {})
    labels_by_code = {int(float(k)): str(v) for k, v in country_labels.items()}
    country_result = sav_country.merge(csv_country, on="country_code", suffixes=("_sav", "_csv"), validate="one_to_one")
    country_result.insert(1, "country", country_result["country_code"].map(labels_by_code))
    assert len(country_result) == EXPECTED_COUNTRIES, "SAV/CSV country sets are inconsistent"
    assert (country_result["n_respondents_sav"] == country_result["n_respondents_csv"]).all()
    assert np.allclose(country_result["sum_weight_sav"], country_result["sum_weight_csv"],
                       atol=SUM_TOLERANCE, rtol=0)
    country_result["cross_source_match"] = (
        (country_result["n_respondents_sav"] == country_result["n_respondents_csv"])
        & np.isclose(country_result["sum_weight_sav"], country_result["sum_weight_csv"],
                     atol=SUM_TOLERANCE, rtol=0)
    )

    print("\nGlobal SAV results:")
    print(sav_total.to_string(index=False, float_format=lambda x: f"{x:.10f}"))
    print("\nCountry weight sums (SAV):")
    print(country_result[["country", "n_respondents_sav", "sum_weight_sav", "weight_minus_n_sav", "normalised_sav"]]
          .to_string(index=False, float_format=lambda x: f"{x:.9f}"))
    print(f"\nSAV/CSV comparison by respondent ID: {n_country_diff} country mismatches, "
          f"{n_weight_diff} weight mismatches; max absolute weight difference={max_weight_diff:.3g}")
    print(f"Weight range: {float(sav_total.loc[0, 'min_weight']):.10f} "
          f"to {float(sav_total.loc[0, 'max_weight']):.10f} "
          "(approximately 0.06–5.89)")
    print("All 24 countries have weight sum = sample size within tolerance 1e-6.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.concat([sav_total, csv_total], ignore_index=True)
    country_result.to_csv(OUTPUT / "task09_weight_country_audit.csv", index=False)
    summary.to_csv(OUTPUT / "task09_weight_summary.csv", index=False)
    note = (
        "TASK 9: Survey weights\n"
        "Decision: Retain the original general-population weight unchanged; use weight for population-attitude estimates.\n"
        "Alternative: unweighted analysis or rescaling the already-normalised weights (not chosen).\n"
        "Evidence: 0 missing/nonpositive weights; within every country sum(weights)=n within 1e-6.\n"
        f"Minimum weight={sav_total.loc[0, 'min_weight']:.12f}; maximum={sav_total.loc[0, 'max_weight']:.12f}.\n"
        f"SAV/CSV respondent-weight mismatches={n_weight_diff}; max absolute difference={max_weight_diff:.12g}.\n"
        "AI use: ChatGPT generated and CSV-tested the validation code; original SAV must be verified independently.\n"
        "Potential AI error: if this script fails, do not alter weights to fit expectations; investigate and log discrepancy.\n"
    )
    (OUTPUT / "task09_validation_note.txt").write_text(note, encoding="utf-8")
    print("\nSaved: task09_weight_country_audit.csv, task09_weight_summary.csv, task09_validation_note.txt")
    print("TASK 9 PASSED: no missing/nonpositive weights; all country sums match sample sizes; SAV and CSV agree.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run tests using synthetic data")
    parser.add_argument("--csv-only", action="store_true", help="Check CSV reference only (not final validation)")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main(csv_only=args.csv_only)
