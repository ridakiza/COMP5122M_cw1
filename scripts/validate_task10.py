"""COMP5122M P1 — Task 10: clean the Q30/Q31 nonresponses, verify every change.

Run from the repository root (same location as Tasks 2–9):
    python scripts/validate_task10.py --self-test
    python scripts/validate_task10.py

Method:
* Load original .sav with numeric response codes and user_missing=True.
* INFER nonresponse codes from each question's own SAV response labels,
  using the label classifier verified in Task 4, NOT by assuming all 8/9
  codes in the survey are missing.
* Make a COPY and recode only Q30 (ai_heard) / Q31 (ai_cncexc).
* Audit response counts before and after; verify against the original CSV
  by respondent ID; ensure all legitimate 1/2/3 answers are unchanged.
* Never overwrite data/raw files or the shared data/clean/pew_clean.csv.

The fixed expected counts below are independent CHECKS (from the supplied
original CSV), not recoding rules. If a count differs, investigate and log it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Reuse the exact label rule used in Task 4 (our validate_task04.py):
# a code is "no answer" if its label says Don't know or Refused. "Not sure" is NOT matched.
from validate_task04 import NONRESPONSE


def is_nonresponse(label: object) -> bool:
    return bool(NONRESPONSE.search(str(label)))

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "clean"
SAV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset.sav"
CSV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv"

QUESTIONS = {"ai_heard": "Q30", "ai_cncexc": "Q31"}
ID_COLS = ["id", "country"]
EXPECTED_N = 28_333
EXPECTED_COUNTRIES = 24
EXPECTED_COUNTS = {
    "ai_heard": {1: 9407, 2: 12878, 3: 5240, 8: 785, 9: 23},
    "ai_cncexc": {1: 4708, 2: 9252, 3: 12409, 8: 1862, 9: 102},
}
EXPECTED_AFTER = {"ai_heard": 27_525, "ai_cncexc": 26_369}


def raw_counts(series: pd.Series) -> dict[int, int]:
    """Counts of each actual numeric response code; don't hide code 8/9."""
    x = pd.to_numeric(series, errors="raise")
    observed = x.dropna()
    assert np.isfinite(observed.to_numpy(dtype=float)).all(), "Non-finite response code"
    assert observed.eq(np.floor(observed)).all(), "Non-integer response code"
    out = x.value_counts(dropna=True).sort_index()
    return {int(k): int(v) for k, v in out.items()}


def metadata_codes(meta: object, variable: str) -> tuple[dict[float, str], set[float]]:
    """Read numeric code/label pairs from SAV; infer nonresponse by TEXT."""
    all_labels = getattr(meta, "variable_value_labels", {}) or {}
    matches = [k for k in all_labels if str(k).strip().casefold() == variable.casefold()]
    assert len(matches) == 1, f"SAV metadata: expected one label map for {variable}; found {matches}"
    raw_labels = all_labels[matches[0]]
    assert raw_labels, f"SAV metadata: {variable} has no labels"
    labels = {float(code): str(label) for code, label in raw_labels.items()}
    nonresponse = {code for code, label in labels.items() if is_nonresponse(label)}
    return labels, nonresponse


def check_frame(df: pd.DataFrame, label: str, expected_n: int = EXPECTED_N) -> None:
    needed = set(ID_COLS + list(QUESTIONS))
    missing = needed - set(df.columns)
    assert not missing, f"{label}: missing fields {sorted(missing)}"
    assert len(df) == expected_n, f"{label}: expected {expected_n} respondents, found {len(df)}"
    assert df["id"].notna().all() and df["id"].is_unique, f"{label}: IDs missing or duplicated"
    for var in QUESTIONS:
        counts = raw_counts(df[var])
        assert set(counts).issubset({1, 2, 3, 8, 9}), (
            f"{label}: {var} contains unexpected values: {set(counts) - {1, 2, 3, 8, 9}}"
        )


def clean_questions(df: pd.DataFrame, meta: object,
                    *, verify_reference: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return NEW cleaned frame and an audit table; do not mutate ``df``."""
    check_frame(df, "INPUT", EXPECTED_N if verify_reference else len(df))
    before = df[ID_COLS + list(QUESTIONS)].copy(deep=True)
    clean = df.copy(deep=True)
    summary = []

    for var, question in QUESTIONS.items():
        labels, bad_codes = metadata_codes(meta, var)
        # Verification expectation from Q30/Q31 questionnaire; does NOT
        # determine the codes used by mask(), which come from SAV labels.
        assert bad_codes == {8.0, 9.0}, (
            f"{var}: SAV labels imply {sorted(bad_codes)}, expected [8, 9]. "
            "STOP and review metadata rather than forcing the recode."
        )
        observed = raw_counts(df[var])
        if verify_reference:
            assert observed == EXPECTED_COUNTS[var], (
                f"{var}: source responses differ from independent CSV reference: "
                f"observed={observed}, expected={EXPECTED_COUNTS[var]}"
            )
        numeric = pd.to_numeric(df[var], errors="raise")
        original_present = int(numeric.notna().sum())
        original_missing = int(numeric.isna().sum())
        removed = int(numeric.isin(bad_codes).sum())
        clean[var] = numeric.mask(numeric.isin(bad_codes), np.nan)
        after_present = int(clean[var].notna().sum())
        after_missing = int(clean[var].isna().sum())

        # Exact changes only; ordinary substantive responses must be preserved.
        assert after_present == original_present - removed
        assert after_missing == original_missing + removed
        assert not clean[var].isin(bad_codes).any()
        for value in (1, 2, 3):
            response_mask = numeric.eq(value)
            assert clean.loc[response_mask, var].eq(value).all(), f"{var}: valid code {value} changed"
        if verify_reference:
            assert after_present == EXPECTED_AFTER[var], (
                f"{var}: after cleaning expected {EXPECTED_AFTER[var]}, got {after_present}"
            )

        print(f"\n{question} — {var}")
        print("  SAV value labels:", {int(k): v for k, v in sorted(labels.items())})
        print("  Nonresponse codes inferred from labels:", sorted(int(x) for x in bad_codes))
        print("  Original code counts:", observed)
        print(f"  Before: {original_present:,} recorded, {original_missing:,} blank")
        print(f"  Recode: {removed:,} to NaN ({observed.get(8, 0):,} code 8; {observed.get(9, 0):,} code 9)")
        print(f"  After:  {after_present:,} valid, {after_missing:,} missing")
        summary.append({
            "question": question, "variable": var, "missing_codes_from_sav_labels": ",".join(str(int(v)) for v in sorted(bad_codes)),
            "code_1_original": observed.get(1, 0), "code_2_original": observed.get(2, 0),
            "code_3_original": observed.get(3, 0), "code_8_original": observed.get(8, 0),
            "code_9_original": observed.get(9, 0), "before_nonnull": original_present,
            "before_blank": original_missing, "converted_to_nan": removed,
            "after_nonnull": after_present, "after_missing": after_missing,
            "n_rows_before": len(df), "n_rows_after": len(clean),
        })

    # Confirm identifiers and respondent order were not touched.
    pd.testing.assert_frame_equal(clean[ID_COLS], before[ID_COLS], check_exact=True)
    pd.testing.assert_frame_equal(df[ID_COLS + list(QUESTIONS)], before, check_exact=True)
    assert len(clean) == len(df) and clean.index.equals(df.index)
    assert clean is not df
    unchanged_columns = df.columns.difference(list(QUESTIONS))
    pd.testing.assert_frame_equal(clean[unchanged_columns], df[unchanged_columns], check_exact=True)
    return clean, pd.DataFrame(summary)


def compare_sav_csv(sav_original: pd.DataFrame, csv_original: pd.DataFrame,
                    sav_clean: pd.DataFrame, csv_clean: pd.DataFrame) -> None:
    """Compare original and recoded values for every ID without assuming row order."""
    cols = ID_COLS + list(QUESTIONS)
    for stage, a, b in [
        ("before cleaning", sav_original, csv_original),
        ("after cleaning", sav_clean, csv_clean),
    ]:
        a = a[cols].set_index("id").sort_index()
        b = b[cols].set_index("id").sort_index()
        assert a.index.equals(b.index), f"{stage}: SAV and CSV have different IDs"
        for col in cols[1:]:
            x = pd.to_numeric(a[col], errors="raise").to_numpy(dtype=float)
            y = pd.to_numeric(b[col], errors="raise").to_numpy(dtype=float)
            equal = np.isclose(x, y, atol=1e-10, rtol=0, equal_nan=True)
            assert bool(equal.all()), f"{stage}: {col} mismatches for {int((~equal).sum())} IDs"
        print(f"PASS: SAV/CSV agree respondent by respondent {stage} ({len(a):,} IDs)")


def country_audit(sav_original: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Extra evidence: number of Q30/Q31 nonresponses removed in each country."""
    parts = []
    for var, question in QUESTIONS.items():
        x = pd.DataFrame({
            "country_code": pd.to_numeric(sav_original["country"], errors="raise").astype(int),
            "was_code_8": pd.to_numeric(sav_original[var], errors="raise").eq(8),
            "was_code_9": pd.to_numeric(sav_original[var], errors="raise").eq(9),
            "valid_after": cleaned[var].notna(),
        })
        grouped = x.groupby("country_code", sort=True).agg(
            n_respondents=("valid_after", "size"),
            n_code_8=("was_code_8", "sum"),
            n_code_9=("was_code_9", "sum"),
            n_valid_after=("valid_after", "sum"),
        ).reset_index()
        grouped.insert(0, "question", question)
        grouped.insert(1, "variable", var)
        grouped["n_missing_after"] = grouped["n_respondents"] - grouped["n_valid_after"]
        parts.append(grouped)
    result = pd.concat(parts, ignore_index=True)
    assert len(result) == 2 * EXPECTED_COUNTRIES
    return result


def self_test() -> None:
    fake_meta = SimpleNamespace(variable_value_labels={
        "ai_heard": {1: "A lot", 2: "A little", 3: "Nothing at all", 8: "Don't know", 9: "Refused"},
        "ai_cncexc": {1: "More excited", 2: "More concerned", 3: "Equally concerned and excited",
                      8: "Don't know", 9: "Refused"},
    })
    toy = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6], "country": [2] * 6,
        "ai_heard": [1, 2, 3, 8, 9, np.nan],
        "ai_cncexc": [9, 8, 3, 2, 1, np.nan],
    })
    snap = toy.copy(deep=True)
    clean, audit = clean_questions(toy, fake_meta, verify_reference=False)
    assert clean["ai_heard"].isna().sum() == 3
    assert clean["ai_cncexc"].isna().sum() == 3
    assert clean.loc[0, "ai_heard"] == 1 and clean.loc[3, "ai_heard"] != 8
    assert clean.loc[3, "ai_cncexc"] == 2
    assert audit["converted_to_nan"].tolist() == [2, 2]
    pd.testing.assert_frame_equal(toy, snap, check_exact=True)
    print("PASS: both Q30/Q31 recode only label-identified nonresponses; original input unchanged")
    assert not is_nonresponse("Not sure")
    assert is_nonresponse("Don't know") and is_nonresponse("Refused")
    print("PASS: valid 'Not sure' is not confused with 'Don't know'")
    wrongly_labelled = SimpleNamespace(variable_value_labels={
        **fake_meta.variable_value_labels,
        "ai_heard": {1: "A lot", 2: "A little", 3: "Nothing at all", 8: "Not sure", 9: "Refused"},
    })
    try:
        clean_questions(toy, wrongly_labelled, verify_reference=False)
    except AssertionError:
        print("PASS: unexpected SAV missing-label definitions are rejected")
    else:
        raise AssertionError("Self-test: an unexpected label change went undetected")
    print("TASK 10 SELF-TEST PASSED")


def main() -> None:
    print("=== TASK 10: Q30/Q31 — RECODE LABELLED NONRESPONSES ===")
    if not SAV.is_file():
        raise FileNotFoundError(f"Missing original SAV: {SAV}")
    if not CSV.is_file():
        raise FileNotFoundError(f"Missing original CSV: {CSV}")
    try:
        import pyreadstat
    except ImportError as exc:
        raise ImportError("Install pyreadstat: python -m pip install pyreadstat") from exc

    raw_df, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    assert raw_df.shape == (EXPECTED_N, 483), f"Unexpected SAV shape: {raw_df.shape}"
    assert raw_df["country"].nunique() == EXPECTED_COUNTRIES
    check_frame(raw_df, "SAV")
    raw_snapshot = raw_df[ID_COLS + list(QUESTIONS)].copy(deep=True)
    print(f"Original SAV: {len(raw_df):,} rows x {raw_df.shape[1]} columns")
    print("Cleaning only ai_heard and ai_cncexc in a NEW DataFrame.")
    cleaned, sav_audit = clean_questions(raw_df, meta)
    pd.testing.assert_frame_equal(raw_df[ID_COLS + list(QUESTIONS)], raw_snapshot, check_exact=True)
    print("PASS: original SAV DataFrame unchanged; all 28,333 people preserved")

    csv_original = pd.read_csv(CSV, usecols=ID_COLS + list(QUESTIONS),
                               low_memory=False, encoding="utf-8-sig")
    check_frame(csv_original, "CSV")
    csv_clean, csv_audit = clean_questions(csv_original, meta)
    compare_sav_csv(raw_df, csv_original, cleaned, csv_clean)
    for field in ("code_1_original", "code_2_original", "code_3_original", "code_8_original",
                  "code_9_original", "before_nonnull", "converted_to_nan", "after_nonnull"):
        assert sav_audit[field].equals(csv_audit[field]), f"SAV/CSV audit mismatch: {field}"

    OUTPUT = OUT
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit_file = OUTPUT / "task10_q30_q31_audit.csv"
    by_country_file = OUTPUT / "task10_q30_q31_country_audit.csv"
    note_file = OUTPUT / "task10_validation_note.txt"
    sav_audit.to_csv(audit_file, index=False)
    country_audit(raw_df, cleaned).to_csv(by_country_file, index=False)
    note_file.write_text(
        "TASK 10 — Q30/Q31 missing-response recoding\n"
        "DECISION: Infer nonresponse codes per variable from SAV value labels; "
        "codes 8 (Don't know) and 9 (Refused) were found in both Q30 and Q31. "
        "Recode these to NaN on a copy. Keep all respondents and all substantive codes.\n"
        "ALTERNATIVE: Treat codes 8/9 as substantive in Q30/Q31 or delete nonrespondents. "
        "Rejected because the questionnaire identifies these as nonresponses; "
        "row deletion would unnecessarily lose respondents.\n"
        "EFFECT: ai_heard: 808 recoded, 27,525 valid; ai_cncexc: 1,964 recoded, "
        "26,369 valid. Zero rows removed. Original source untouched.\n"
        "AI USE: ChatGPT generated the validation/cleaning script and independently "
        "counted response codes using the CSV; running the script verifies the "
        "original SAV, the labels, and person-by-person agreement.\n"
        "CHECK FOR TASK 19: Pew toplines include DK/refused in their denominators. "
        "Do not calculate topline percentages only over non-null cleaned values; "
        "use the total eligible/responding sample weight in the denominator.\n",
        encoding="utf-8",
    )
    print("\nSaved:", audit_file.name, by_country_file.name, note_file.name)
    print("TASK 10 PASSED: Q30/Q31 codes identified from SAV labels, recoded only on a copy;")
    print("                 all 28,333 respondents retained; SAV and CSV results agree.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Test recoding logic on synthetic data")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()