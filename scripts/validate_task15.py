"""COMP5122M P1 — Task 15: resolve Israel's contradictory age information.

Run from the repository root:
    python scripts/validate_task15.py --self-test
    python scripts/validate_task15.py

Checks SAV labels and both original files without changing either one.
Two Israelis with exact ages also have follow-up data; only one is contradictory.
Keep each person's exact age as the primary answer and document the exception.

Respondent-level audit output is PRIVATE: data/clean/ (never commit to GitHub).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "clean"
SAV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset.sav"
CSV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv"
FIELDS = ["id", "country", "age", "d_age_israel"]


def labels_for(meta: object, variable: str) -> dict[int, str]:
    """Read response labels directly from the original SAV metadata."""
    all_labels = getattr(meta, "variable_value_labels", {}) or {}
    matching = [k for k in all_labels if str(k).strip().casefold() == variable]
    assert len(matching) == 1, f"Expected one SAV label dictionary for {variable}, got {matching}"
    return {int(float(code)): str(label).strip()
            for code, label in all_labels[matching[0]].items()}


def interval_from_label(label: str) -> tuple[int, float]:
    """Use stated age limits, never midpoint guesses."""
    s = str(label).strip().replace("–", "-").replace("—", "-")
    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
    if match:
        low, high = map(int, match.groups())
        assert 18 <= low <= high <= 120, f"Invalid follow-up interval {label!r}"
        return low, float(high)
    match = re.fullmatch(r"(\d+)\s*\+", s)
    if match:
        low = int(match.group(1))
        assert 18 <= low <= 120, f"Invalid follow-up interval {label!r}"
        return low, float("inf")
    raise AssertionError(f"Unexpected follow-up age label {label!r}: review the SAV dictionary")


def band_from_exact(age: int) -> str:
    if 18 <= age <= 29:
        return "18-29"
    if 30 <= age <= 49:
        return "30-49"
    if 50 <= age <= 64:
        return "50-64"
    if 65 <= age <= 97:
        return "65+"  # A 97 means 97 OR OLDER; still unambiguously 65+.
    raise AssertionError(f"Not a usable exact age: {age}")


def prepare(raw: pd.DataFrame, source: str, *, strict: bool) -> pd.DataFrame:
    """Copy selected columns and normalise only whitespace blanks for comparison."""
    renamed = {col: str(col).strip().casefold() for col in raw.columns}
    assert len(set(renamed.values())) == len(renamed), f"{source}: duplicate normalised column names"
    assert set(FIELDS).issubset(renamed.values()), f"{source}: required fields absent"
    df = raw.rename(columns=renamed)[FIELDS].copy(deep=True)
    for col in FIELDS:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            text = df[col].astype("string").str.strip()
            df[col] = text.mask(text.eq(""))
        df[col] = pd.to_numeric(df[col], errors="raise")
        assert np.isfinite(df[col].dropna().to_numpy(dtype=float)).all(), f"{source}: nonfinite {col}"
    assert df.id.notna().all() and df.id.is_unique, f"{source}: bad respondent IDs"
    if strict:
        assert len(df) == 28_333 and raw.shape == (28_333, 483), f"{source}: wrong dataset dimensions"
    return df


def compare_by_id(sav: pd.DataFrame, csv: pd.DataFrame) -> None:
    a, b = sav.set_index("id").sort_index(), csv.set_index("id").sort_index()
    assert a.index.equals(b.index), "SAV and CSV IDs differ"
    for col in ["country", "age", "d_age_israel"]:
        x, y = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        if not np.isclose(x, y, atol=0, rtol=0, equal_nan=True).all():
            raise AssertionError(f"SAV/CSV respondent-level differences in {col}")
    print(f"PASS: SAV/CSV agree person by person on {len(a):,} IDs for country, age, d_age_israel")


def audit_israel(df: pd.DataFrame, meta: object, *, strict: bool) -> pd.DataFrame:
    """Return the cases with an age follow-up despite also having an exact age."""
    countries = labels_for(meta, "country")
    isr = [code for code, name in countries.items() if name.strip().casefold() == "israel"]
    assert len(isr) == 1, f"SAV country metadata does not uniquely identify Israel: {isr}"
    country_code = isr[0]
    assert df.loc[df.country.ne(country_code), "d_age_israel"].isna().all(), (
        "Israeli follow-up recorded outside Israel"
    )
    followup_labels = labels_for(meta, "d_age_israel")
    age_labels = labels_for(meta, "age")
    assert "97" in age_labels[97] and "older" in age_labels[97].casefold()
    assert "know" in age_labels[98].casefold() and "refus" in age_labels[99].casefold()

    mask_isr = df.country.eq(country_code)
    isr_followups = df.loc[mask_isr & df.d_age_israel.notna()].copy()
    assert isr_followups.d_age_israel.mod(1).eq(0).all(), "Noninteger follow-up code"
    unknown_codes = set(isr_followups.d_age_israel.astype(int)) - set(followup_labels)
    assert not unknown_codes, f"Unknown follow-up response codes: {unknown_codes}"

    records = []
    for row in isr_followups.itertuples(index=False):
        # This task concerns FOLLOW-UPS WHERE AN EXACT AGE ALREADY EXISTS.
        if pd.isna(row.age) or not (18 <= row.age <= 97):
            continue
        assert float(row.age).is_integer(), f"Invalid exact age: {row.age}"
        code = int(row.d_age_israel)
        label = followup_labels[code]
        low, high = interval_from_label(label)
        consistent = low <= row.age <= high
        records.append({
            "id": int(row.id),
            "country": "Israel",
            "exact_age": int(row.age),
            "followup_code": code,
            "followup_label": label,
            "followup_low": int(low),
            "followup_high": int(high) if np.isfinite(high) else "open-ended",
            "answers_consistent": bool(consistent),
            "issue_type": "unexpected_followup_but_consistent" if consistent else "conflicting_age_answers",
            "age_band_from_exact": band_from_exact(int(row.age)),
            "decision": "keep_exact_age_and_its_age_band",
        })
    out = pd.DataFrame(records).sort_values("id").reset_index(drop=True)
    if strict:
        assert len(out) == 2, f"Expected two follow-ups despite exact ages; found {len(out)}"
        assert out.answers_consistent.sum() == 1, "Expected exactly one consistent follow-up"
        assert (~out.answers_consistent).sum() == 1, "Expected exactly one true contradiction"
        assert set(out.exact_age) == {63, 65}, "Unexpected exact ages in Israeli records"
        assert set(zip(out.exact_age, out.age_band_from_exact)) == {(63, "50-64"), (65, "65+")}
        assert out.followup_low.eq(45).all() and out.followup_high.eq(64).all()
    return out


def self_test() -> None:
    meta = SimpleNamespace(variable_value_labels={
        "country": {13: "Australia", 18: "Israel"},
        "age": {97: "97 or older", 98: "Don't know", 99: "Refused"},
        "d_age_israel": {3: "45-64", 4: "65+", 9: "Refused"},
    })
    data = pd.DataFrame({
        "id": [100, 101, 102], "country": [18, 18, 13],
        "age": [63, 65, np.nan], "d_age_israel": [3, 3, np.nan],
    })
    original = data.copy(deep=True)
    result = audit_israel(prepare(data, "SYNTHETIC", strict=False), meta, strict=True)
    assert result["answers_consistent"].tolist() == [True, False]
    assert result["age_band_from_exact"].tolist() == ["50-64", "65+"]
    pd.testing.assert_frame_equal(data, original)
    print("PASS: age 63 is consistent with 45-64; age 65 contradicts 45-64")
    print("PASS: exact ages yield bands 50-64 and 65+; original records unchanged")

    wrong = data.copy(deep=True)
    wrong.loc[0, "d_age_israel"] = 42
    try:
        audit_israel(prepare(wrong, "SYNTHETIC", strict=False), meta, strict=False)
    except AssertionError as exc:
        assert "Unknown follow-up" in str(exc)
        print("PASS: undocumented follow-up codes are rejected")
    else:
        raise AssertionError("Synthetic unknown response code was not detected")

    bad_csv = data.copy(deep=True)
    bad_csv.loc[1, "age"] = 64
    try:
        compare_by_id(prepare(data, "SAV", strict=False),
                      prepare(bad_csv, "CSV", strict=False))
    except AssertionError as exc:
        assert "respondent-level" in str(exc)
        print("PASS: deliberate SAV/CSV disagreement is detected")
    else:
        raise AssertionError("Synthetic CSV inconsistency was not detected")
    print("TASK 15 SELF-TEST PASSED")


def main() -> None:
    import pyreadstat

    for path in (SAV, CSV):
        if not path.is_file():
            raise FileNotFoundError(f"Missing original source: {path}")
    print("=== TASK 15: ISRAEL AGE-FOLLOW-UP CONFLICT ===")
    sav_raw, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    csv_raw = pd.read_csv(CSV, low_memory=False, encoding="utf-8-sig")
    sav_names = {str(col).strip().casefold(): col for col in sav_raw.columns}
    csv_names = {str(col).strip().casefold(): col for col in csv_raw.columns}
    sav_original_cols = [sav_names[name] for name in FIELDS]
    csv_original_cols = [csv_names[name] for name in FIELDS]
    sav_source_copy = sav_raw[sav_original_cols].copy(deep=True)
    csv_source_copy = csv_raw[csv_original_cols].copy(deep=True)
    sav = prepare(sav_raw, "SAV", strict=True)
    csv = prepare(csv_raw, "CSV", strict=True)
    compare_by_id(sav, csv)

    a = audit_israel(sav, meta, strict=True)
    b = audit_israel(csv, meta, strict=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    pd.testing.assert_frame_equal(sav_raw[sav_original_cols], sav_source_copy)
    pd.testing.assert_frame_equal(csv_raw[csv_original_cols], csv_source_copy)
    assert len(sav_raw) == len(csv_raw) == 28_333

    print("SAV age follow-up label:", labels_for(meta, "d_age_israel")[3])
    print("\nCases where a follow-up was recorded even though exact age exists:")
    print(a[["exact_age", "followup_label", "answers_consistent",
             "issue_type", "age_band_from_exact", "decision"]].to_string(index=False))
    n_consistent = int(a.answers_consistent.sum())
    n_contradictions = int((~a.answers_consistent).sum())
    print(f"\nUnexpected follow-ups with exact ages: {len(a)}")
    print(f"  Consistent with exact age: {n_consistent}")
    print(f"  Genuinely contradictory:  {n_contradictions}")
    print("Decision: retain ages 63 and 65; retain respective bands 50-64 and 65+.")
    print("PASS: no age was overwritten; no respondents removed; SAV/CSV cases identical")

    OUT.mkdir(parents=True, exist_ok=True)
    a.to_csv(OUT / "task15_israel_age_conflicts.csv", index=False)
    pd.DataFrame([{
        "unexpected_followups": len(a),
        "consistent_followups": n_consistent,
        "contradictory_followups": n_contradictions,
        "ages_overwritten": 0,
        "respondents_removed": 0,
        "age_bands_retained_from_exact_age": len(a),
    }]).to_csv(OUT / "task15_israel_age_summary.csv", index=False)
    pd.DataFrame([{
        "task": 15,
        "decision": "For Israeli respondents with both an exact age and a follow-up, use the exact age",
        "alternative": "Use the country age-follow-up (45-64) or drop both ambiguous records",
        "reason": "Direct exact-age response is more specific; a follow-up is intended for missing ages",
        "effect": "1 genuine contradiction flagged; ages 63 and 65 retained in bands 50-64 and 65+; no change to Task 14 recovery counts",
        "respondents_affected": len(a),
        "respondents_removed": 0,
        "verification": "Original SAV labels, respondent-level SAV/CSV equality, independent Task 14 results",
        "status": "verified_by_SAV_CSV_run",
    }]).to_csv(OUT / "task15_decision_log_entry.csv", index=False)
    pd.DataFrame([{
        "task": 15,
        "ai_contribution": "Designed exact-age precedence rule and generated validation/audit code",
        "independent_evidence": "Pew SAV labels, original CSV and previous Task 14 country follow-up audit",
        "human_verification": "User executes SAV/CSV cross-check and reviews output",
        "ai_error_or_clarification": "Terminology clarified: 2 routing anomalies, only 1 contradictory age answer",
        "correction": "Count age 63 as consistent-but-unexpected; age 65 as genuine contradiction",
    }]).to_csv(OUT / "task15_ai_use_log_entry.csv", index=False)
    (OUT / "task15_validation_note.txt").write_text(
        "TASK 15: Two Israeli age follow-up responses exist despite an exact age. "
        "Exact age 63 agrees with follow-up 45-64; exact age 65 contradicts it. "
        "Preserve the original exact ages, deriving 50-64 and 65+ respectively. "
        "No imputation, no deletions, no original data changes. "
        "The audit with respondent IDs is private and should not be committed.\n",
        encoding="utf-8",
    )
    print("\nSaved: task15_israel_age_conflicts.csv, task15_israel_age_summary.csv,")
    print("       task15_decision_log_entry.csv, task15_ai_use_log_entry.csv,")
    print("       task15_validation_note.txt")
    print("TASK 15 PASSED: 2 unexpected follow-ups, 1 contradictory; exact ages retained.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
