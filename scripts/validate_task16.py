"""COMP5122M P1 — Task 16: make one unified, auditable age_band variable.

From the repository ROOT:
    python scripts/validate_task16.py --self-test
    python scripts/validate_task16.py

Reuses Task 14's VERIFIED whole-interval age recovery. Applies Australia's
own SAV-labelled age bands to Australian respondents ONLY. Does not impute
exact ages, guess overlapping intervals, drop rows, or overwrite pew_clean.csv.
Outputs (including respondent IDs) are PRIVATE under data/clean/.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# python scripts/validate_task16.py makes scripts/ importable.
from validate_task14 import (  # type: ignore
    BANDS,
    country_age_columns,
    metadata_labels,
    normal_label,
    prepare,
    recover,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "clean"
SAV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset.sav"
CSV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv"
ORDER = [name for name, _, _ in BANDS]
EXPECTED = {
    "respondents": 28_333,
    "australian_respondents": 1_829,
    "australian_reported": 1_827,
    "australian_missing": 2,
    "exact_age": 26_210,
    "recovered_followup": 111,
    "missing_other_countries": 183,
    "all_age_known": 28_148,
    "all_age_missing": 185,
    "by_band": {"18-29": 6_068, "30-49": 9_787, "50-64": 6_596, "65+": 5_697},
    "aus_by_band": {"18-29": 326, "30-49": 658, "50-64": 443, "65+": 400},
}


def australian_age_mapping(meta: object) -> dict[int, str]:
    """Read code-to-age-band meanings from original SAV metadata, not guessed codes."""
    labels = metadata_labels(meta, "age_australia")
    result: dict[int, str] = {}
    for code, label in labels.items():
        text = re.sub(r"\s+", "", str(label)).replace("–", "-").replace("—", "-")
        assert text in ORDER, f"Unexpected Australian age label {code}={label!r}"
        result[code] = text
    assert set(result.values()) == set(ORDER) and len(result) == 4, (
        f"Unexpected Australian categories in SAV: {result}"
    )
    return result


def make_age_band(partial: pd.DataFrame, meta: object, *, strict: bool = True) -> pd.DataFrame:
    """Add age_band to Task14 result, preserving recovered bands and source provenance."""
    needed = {"id", "country", "age", "age_clean", "age_australia", "age_band_partial", "age_band_source"}
    assert needed.issubset(partial.columns), f"Missing Task14 fields: {needed - set(partial.columns)}"
    assert partial.id.notna().all() and partial.id.is_unique
    before = partial.copy(deep=True)
    countries = metadata_labels(meta, "country")
    australian_codes = [code for code, name in countries.items() if normal_label(name) == "australia"]
    assert len(australian_codes) == 1, "Cannot uniquely identify Australia in SAV country labels"
    aus = partial.country.eq(australian_codes[0])
    assert partial.loc[~aus, "age_australia"].isna().all(), "Australia ages entered outside Australia"
    assert partial.loc[aus, "age"].isna().all(), "Exact age unexpectedly populated for Australians"
    assert partial.loc[aus, "age_clean"].isna().all(), "Australian exact age unexpectedly recovered"
    assert partial.loc[aus, "age_band_partial"].isna().all(), "Task14 unexpectedly prefilled Australia"
    assert partial.loc[~aus, "age_band_source"].isin(
        ["exact_age", "recovered_followup", "unclassified"]
    ).all(), "Unexpected Task14 source outside Australia"

    code_map = australian_age_mapping(meta)
    present = partial.loc[aus, "age_australia"].dropna()
    assert np.isfinite(present.to_numpy(dtype=float)).all()
    assert present.mod(1).eq(0).all(), "Non-integer Australian age code"
    unknown = set(present.astype(int).unique()) - set(code_map)
    assert not unknown, f"Unlabelled Australian age codes: {unknown}"

    out = partial.copy(deep=True)
    out["age_band"] = out["age_band_partial"].astype("string")
    out.loc[aus, "age_band"] = out.loc[aus, "age_australia"].map(code_map).astype("string")
    out["age_band_source"] = out["age_band_source"].astype("string")
    out.loc[aus & out.age_band.notna(), "age_band_source"] = "australia_reported"
    out.loc[aus & out.age_band.isna(), "age_band_source"] = "australia_missing"

    assert out.loc[~aus, "age_band"].fillna("<NA>").equals(
        before.loc[~aus, "age_band_partial"].astype("string").fillna("<NA>")
    ), "Pre-existing Task14 age bands were changed"
    assert out.loc[~aus, "age_band_source"].astype(str).equals(
        before.loc[~aus, "age_band_source"].astype(str)
    ), "Non-Australian provenance was modified"
    assert out.loc[out.age_clean.eq(97), "age_band"].eq("65+").all(), "Top-coded 97+ not retained"
    assert not out.loc[out.age_band.notna(), "age_band"].isin(ORDER).eq(False).any()
    assert len(out) == len(before) and out.id.equals(before.id)
    pd.testing.assert_frame_equal(partial, before, check_exact=True)

    if strict:
        counts = out.age_band.value_counts().reindex(ORDER, fill_value=0).to_dict()
        sources = out.age_band_source.value_counts().to_dict()
        assert len(out) == EXPECTED["respondents"]
        assert counts == EXPECTED["by_band"], f"Unexpected overall age groups: {counts}"
        assert int(aus.sum()) == EXPECTED["australian_respondents"]
        assert int((aus & out.age_band.notna()).sum()) == EXPECTED["australian_reported"]
        assert int((aus & out.age_band.isna()).sum()) == EXPECTED["australian_missing"]
        assert out.loc[aus, "age_band"].value_counts().to_dict() == EXPECTED["aus_by_band"]
        assert int(out.age_band.notna().sum()) == EXPECTED["all_age_known"]
        assert int(out.age_band.isna().sum()) == EXPECTED["all_age_missing"]
        assert int((~aus & out.age_band.isna()).sum()) == EXPECTED["missing_other_countries"]
        assert sources == {
            "exact_age": EXPECTED["exact_age"],
            "recovered_followup": EXPECTED["recovered_followup"],
            "australia_reported": EXPECTED["australian_reported"],
            "australia_missing": EXPECTED["australian_missing"],
            "unclassified": EXPECTED["missing_other_countries"],
        }, f"Unexpected sources: {sources}"
        # The two Task15 Israeli cases must still use exact reported ages.
        assert set(out.loc[out.age_clean.eq(63), "age_band"].dropna()) <= {"50-64"}
        assert set(out.loc[out.age_clean.eq(65), "age_band"].dropna()) <= {"65+"}
    return out


def check_person_by_person(sav: pd.DataFrame, csv: pd.DataFrame) -> None:
    left = sav.set_index("id").sort_index()
    right = csv.set_index("id").sort_index()
    assert left.index.equals(right.index), "SAV/CSV respondent IDs differ"
    fields = ["age_band", "age_band_source", "age_band_partial", "age_clean"]
    for col in fields:
        a, b = left[col], right[col]
        if col == "age_clean":
            ok = np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                            equal_nan=True, rtol=0, atol=0).all()
        else:
            ok = a.astype("string").fillna("<NA>").eq(
                b.astype("string").fillna("<NA>")
            ).all()
        assert bool(ok), f"SAV/CSV age classification mismatch for {col}"
    print(f"PASS: SAV/CSV age bands agree person by person across {len(left):,} IDs")


def self_test() -> None:
    from types import SimpleNamespace
    meta = SimpleNamespace(variable_value_labels={
        "country": {2: "Canada", 13: "Australia"},
        "age_australia": {1: "18-29", 2: "30-49", 3: "50-64", 4: "65+"},
    })
    toy = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "country": [2, 2, 2, 2, 13, 13, 13, 13, 13],
        "age": [29, 30, 97, 99, np.nan, np.nan, np.nan, np.nan, np.nan],
        "age_clean": [29, 30, 97, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "age_australia": [np.nan]*4 + [1, 2, 3, 4, np.nan],
        "age_band_partial": ["18-29", "30-49", "65+", "50-64", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
        "age_band_source": ["exact_age", "exact_age", "exact_age", "recovered_followup"] + ["australia_pending_task16"]*5,
    })
    old = toy.copy(deep=True)
    result = make_age_band(toy, meta, strict=False)
    assert result.age_band.tolist()[:8] == ["18-29", "30-49", "65+", "50-64", "18-29", "30-49", "50-64", "65+"]
    assert pd.isna(result.at[8, "age_band"])
    assert result.at[8, "age_band_source"] == "australia_missing"
    pd.testing.assert_frame_equal(toy, old, check_exact=True)
    print("PASS: correct ages, recovered age band, all four Australian bands and blank preserved")
    print("PASS: original input unchanged; exact age 97+ correctly classified as 65+")

    wrong = toy.copy(deep=True)
    wrong.loc[4, "age_australia"] = 98
    try:
        make_age_band(wrong, meta, strict=False)
    except AssertionError as err:
        assert "Unlabelled Australian age codes" in str(err)
        print("PASS: undocumented Australian age code rejected")
    else:
        raise AssertionError("Self-test failed to detect unknown Australian age code")

    wrong = toy.copy(deep=True)
    wrong.loc[0, "age_australia"] = 1
    try:
        make_age_band(wrong, meta, strict=False)
    except AssertionError as err:
        assert "Australia ages entered outside" in str(err)
        print("PASS: Australian age code assigned to wrong country rejected")
    else:
        raise AssertionError("Self-test failed to detect incorrect country routing")
    print("TASK 16 SELF-TEST PASSED")


def main() -> None:
    import pyreadstat
    for p in (SAV, CSV):
        if not p.is_file():
            raise FileNotFoundError(f"Missing original source: {p}")
    print("=== TASK 16: UNIFIED AGE BANDS (INCLUDING AUSTRALIA) ===")
    sav_raw, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    csv_raw = pd.read_csv(CSV, low_memory=False, encoding="utf-8-sig")
    assert sav_raw.shape == csv_raw.shape == (28_333, 483)
    countries, mapping = country_age_columns(meta)
    fields = ["id", "country", "age", "age_australia"] + sorted(mapping.values())
    sav = prepare(sav_raw, fields, "SAV")
    csv = prepare(csv_raw, fields, "CSV")
    # Task14 independently processes SAV and CSV using the full follow-up metadata.
    partial_sav, audit_sav, conflicts_sav = recover(sav, meta)
    partial_csv, audit_csv, conflicts_csv = recover(csv, meta)
    pd.testing.assert_frame_equal(audit_sav, audit_csv)
    pd.testing.assert_frame_equal(conflicts_sav, conflicts_csv, check_dtype=False)
    full_sav = make_age_band(partial_sav, meta)
    full_csv = make_age_band(partial_csv, meta)
    check_person_by_person(full_sav, full_csv)

    aus_code = next(c for c,n in countries.items() if normal_label(n) == "australia")
    aus = full_sav.country.eq(aus_code)
    bands = full_sav.age_band.value_counts().reindex(ORDER, fill_value=0)
    aus_bands = full_sav.loc[aus, "age_band"].value_counts().reindex(ORDER, fill_value=0)
    other_bands = full_sav.loc[~aus, "age_band"].value_counts().reindex(ORDER, fill_value=0)
    summary = pd.DataFrame({
        "age_band": ORDER + ["Missing", "Total"],
        "outside_australia": list(other_bands.astype(int)) + [int(full_sav.loc[~aus,"age_band"].isna().sum()), int((~aus).sum())],
        "australia": list(aus_bands.astype(int)) + [int(full_sav.loc[aus,"age_band"].isna().sum()), int(aus.sum())],
        "total": list(bands.astype(int)) + [int(full_sav.age_band.isna().sum()), len(full_sav)],
    })
    assert summary.total.iloc[:-1].sum() == EXPECTED["respondents"]
    print("Australian SAV age labels:", australian_age_mapping(meta))
    print("\nUnified age bands:")
    print(summary.to_string(index=False))
    print("\nSource counts:")
    print(full_sav.age_band_source.value_counts().to_string())
    print("\nCHECK: 111 recovered age bands, 183 other-country missing + 2 Australia missing = 185 total")
    print("PASS: 97+ preserved; Israeli exact ages preserved; raw input unchanged; no invented exact ages")

    country_summary = []
    for code, name in countries.items():
        subset = full_sav.loc[full_sav.country.eq(code)]
        row = {"country_code": code, "country": name, "n": len(subset),
               "missing_age_band": int(subset.age_band.isna().sum())}
        row.update({"n_"+band: int(subset.age_band.eq(band).sum()) for band in ORDER})
        row["recovered_followup"] = int(subset.age_band_source.eq("recovered_followup").sum())
        country_summary.append(row)
    country_audit = pd.DataFrame(country_summary)
    assert country_audit.n.sum() == EXPECTED["respondents"]
    assert country_audit.missing_age_band.sum() == EXPECTED["all_age_missing"]

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "task16_age_band_summary.csv", index=False)
    country_audit.to_csv(OUT / "task16_age_band_country_audit.csv", index=False)
    full_sav[["id", "country", "age_clean", "age_australia", "age_band", "age_band_source"]].to_csv(
        OUT / "task16_age_band_preview.csv", index=False
    )
    pd.DataFrame([{
        "task":16,
        "decision":"Combine exact ages, unambiguous country follow-up bands, and Australia-labelled age bands",
        "alternative":"Omit Australia, or guess the 183 ambiguous/unanswered follow-ups or 2 blank Australian ages",
        "reason":"Use observed responses and SAV labels; avoid inventing age categories",
        "effect":"28,148 classified, 185 missing (183 outside Australia and 2 in Australia); 0 people removed",
        "respondents_removed":0,
        "verification":"SAV and CSV matched by ID; 24-country audit; original Task14 recovery reproduced",
        "status":"pending_user_SAV_run",
    }]).to_csv(OUT / "task16_decision_log_entry.csv", index=False)
    pd.DataFrame([{
        "task":16,
        "ai_contribution":"Created unified age-band validation code, independent CSV/dictionary reference totals and tests",
        "independent_evidence":"Original Pew CSV, data dictionary and Task14 source classifications",
        "human_verification":"Pending execution against original SAV",
        "verification_outcome":"pending_user_verification",
        "ai_errors_identified":"Previous generic 183-missing statement omitted two Australian blanks",
        "correction":"Use 185 missing overall; 183 refers only to non-Australian respondents",
    }]).to_csv(OUT / "task16_ai_use_log_entry.csv", index=False)
    (OUT / "task16_validation_note.txt").write_text(
        "Task 16: 28,148 classified respondents, 185 age bands unavailable. "
        "Source breakdown: 26,210 exact ages, 111 recovered country follow-up groups, "
        "1,827 Australian age categories; missing: 183 elsewhere, 2 Australia. "
        "Original SAV/CSV untouched; no rows or exact ages imputed. "
        "Record earlier blanket 183-missing statement as an AI ambiguity corrected to 185 globally.\n",
        encoding="utf-8",
    )
    print("\nSaved: task16_age_band_summary.csv, task16_age_band_country_audit.csv,")
    print("       task16_age_band_preview.csv, task16_decision_log_entry.csv,")
    print("       task16_ai_use_log_entry.csv, task16_validation_note.txt")
    print("TASK 16 PASSED: all 28,333 respondents retained; 28,148 age bands known, 185 missing;")
    print("                SAV and CSV agree on each person's classification.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
