"""COMP5122M P1 — Task 17: merge country-specific gender fields without losing 'Other'.

Run from the repository ROOT:
    python scripts/validate_task17.py --self-test
    python scripts/validate_task17.py

Gender meanings come from original SAV value labels (NOT hard-coded numeric
response assumptions).  The expected counts below are validation targets
independently calculated from the supplied CSV, not cleaning rules.

This is an isolated validation: it never overwrites data/clean/pew_clean.csv.
Outputs in data/clean/ must remain PRIVATE while the repository is public.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "clean"
SAV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset.sav"
CSV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv"

EXPECTED = {"Male": 15_107, "Female": 13_203, "Other": 21, "Missing": 2}
LABEL_TO_OUTPUT = {"male": (1, "Male"), "female": (2, "Female"), "other": (3, "Other")}


def metadata_labels(meta: object, variable: str) -> dict[int, str]:
    """Locate the variable's numeric-code labels in the original SAV metadata."""
    mappings = {
        str(name).strip().lower(): value
        for name, value in meta.variable_value_labels.items()
    }
    if variable not in mappings:
        raise AssertionError(f"SAV value labels missing for {variable}")
    labels = {}
    for code, label in mappings[variable].items():
        numeric = float(code)
        assert numeric.is_integer(), f"Non-integer SAV code in {variable}: {code!r}"
        labels[int(numeric)] = str(label)
    return labels


def norm(value: object) -> str:
    return " ".join(str(value).casefold().strip().split())


def is_nonresponse(label: str) -> bool:
    """Detect nonresponse semantics by label, not arbitrary code numbers."""
    value = norm(label).replace("’", "'")
    return (
        "refused" in value
        or "don't know" in value
        or "do not know" in value
        or "prefer not to say" in value
        or "prefer not to answer" in value
        or "no answer" in value
    )


def codebook_mapping(meta: object) -> tuple[dict[str, dict[int, int | None]], dict]:
    """Build code-to-unified-code maps using *SAV label meaning*.

    Unrecognised labels are rejected (not silently converted to missing).
    """
    mappings: dict[str, dict[int, int | None]] = {}
    original: dict[str, dict[int, str]] = {}
    for field in ("gender", "gender_australia"):
        labels = metadata_labels(meta, field)
        original[field] = labels
        mapped: dict[int, int | None] = {}
        for code, label in labels.items():
            key = norm(label)
            if key in LABEL_TO_OUTPUT:
                mapped[code] = LABEL_TO_OUTPUT[key][0]
            elif is_nonresponse(label):
                mapped[code] = None
            else:
                raise AssertionError(f"Unexpected SAV gender label: {field}={code}: {label!r}")
        mappings[field] = mapped

    male_female = {1, 2}
    assert {v for v in mappings["gender"].values() if v is not None} == male_female, (
        f"Unexpected non-Australian gender labels: {original['gender']}"
    )
    assert {v for v in mappings["gender_australia"].values() if v is not None} == {1, 2, 3}, (
        f"Unexpected Australian gender labels: {original['gender_australia']}"
    )
    return mappings, original


def numeric(series: pd.Series) -> pd.Series:
    """Preserve system blanks, including whitespace-only blanks in the CSV."""
    if pd.api.types.is_numeric_dtype(series):
        result = pd.to_numeric(series, errors="raise")
    else:
        text = series.astype("string").str.strip()
        result = pd.to_numeric(text.mask(text.eq("")), errors="raise")
    valid = result.dropna()
    assert np.isfinite(valid.to_numpy(dtype=float)).all(), "Non-finite source code"
    assert valid.mod(1).eq(0).all(), "Non-integer source code"
    return result


def clean_gender(raw: pd.DataFrame, meta: object, *, strict: bool = True) -> pd.DataFrame:
    """Merge gender columns on a new copy; enforce country routing and metadata codes."""
    mandatory = {"id", "country", "gender", "gender_australia"}
    assert mandatory.issubset(raw.columns), f"Missing source columns: {mandatory-set(raw.columns)}"
    before = raw.copy(deep=True)
    assert raw.id.notna().all() and raw.id.is_unique, "Missing or duplicate respondent IDs"

    countries = metadata_labels(meta, "country")
    aus_codes = [code for code, name in countries.items() if norm(name) == "australia"]
    assert len(aus_codes) == 1, f"Cannot uniquely identify Australia from SAV labels: {countries}"
    country = numeric(raw["country"])
    aus = country.eq(aus_codes[0])
    domestic = numeric(raw["gender"])
    australian = numeric(raw["gender_australia"])

    assert domestic.loc[aus].isna().all(), "General gender unexpectedly recorded in Australia"
    assert australian.loc[~aus].isna().all(), "Australian gender recorded outside Australia"
    if strict:
        assert len(raw) == 28_333
        assert int(aus.sum()) == 1_829
        assert int(domestic.loc[~aus].notna().sum()) == 26_504
        assert int(australian.loc[aus].notna().sum()) == 1_827

    mappings, _ = codebook_mapping(meta)
    for field, series in (("gender", domestic), ("gender_australia", australian)):
        observed_codes = set(series.dropna().astype(int).unique())
        unknown = observed_codes - set(mappings[field])
        assert not unknown, f"Unexpected observed codes in {field}: {sorted(unknown)}"

    # Respect country ownership: don't use a value from the wrong field.
    source_codes = domestic.where(~aus, australian)
    code_to_unified = pd.Series(index=raw.index, dtype="Int64")
    code_to_unified.loc[~aus] = domestic.loc[~aus].map(mappings["gender"]).astype("Int64")
    code_to_unified.loc[aus] = australian.loc[aus].map(mappings["gender_australia"]).astype("Int64")

    out = raw.copy(deep=True)
    out["gender_clean"] = code_to_unified
    out["gender_label"] = code_to_unified.map({1: "Male", 2: "Female", 3: "Other"}).astype("string")
    out["gender_source"] = pd.Series(np.where(aus, "australia", "outside_australia"), index=raw.index, dtype="string")
    out.loc[aus & out.gender_clean.isna(), "gender_source"] = "australia_missing"
    assert out["gender_clean"].notna().equals(source_codes.notna() & ~(
        # A labelled refusal is possible but not counted as usable.
        pd.Series(np.where(aus, source_codes.isin([c for c,v in mappings['gender_australia'].items() if v is None]),
                           source_codes.isin([c for c,v in mappings['gender'].items() if v is None])), index=raw.index)
    )), "Missingness during merging doesn't match the source labels"
    assert len(out) == len(raw) and out.id.equals(raw.id)
    pd.testing.assert_frame_equal(raw, before, check_exact=True)
    pd.testing.assert_frame_equal(out[raw.columns], before, check_exact=True)

    if strict:
        actual = out.gender_label.value_counts(dropna=False)
        results = {label: int(actual.get(label, 0)) for label in ("Male", "Female", "Other")}
        results["Missing"] = int(out.gender_label.isna().sum())
        assert results == EXPECTED, f"Unexpected gender distribution: {results}"
        assert int((aus & out.gender_clean.eq(3).fillna(False)).sum()) == 21
        assert int((~aus & out.gender_clean.eq(3).fillna(False)).sum()) == 0
    return out


def compare_source_and_output(sav: pd.DataFrame, csv: pd.DataFrame) -> None:
    """Check respondent IDs AND raw gender/country values AND cleaned results."""
    a = sav.set_index("id").sort_index()
    b = csv.set_index("id").sort_index()
    assert a.index.equals(b.index), "SAV/CSV respondent IDs differ"
    for field in ("country", "gender", "gender_australia", "gender_clean"):
        one, two = numeric(a[field]), numeric(b[field])
        equal = np.isclose(
            one.to_numpy(dtype=float, na_value=np.nan),
            two.to_numpy(dtype=float, na_value=np.nan),
            rtol=0, atol=0, equal_nan=True
        )
        assert bool(equal.all()), f"SAV/CSV mismatch for {field}: {(~equal).sum()} records"
    for field in ("gender_label", "gender_source"):
        assert a[field].astype("string").fillna("<missing>").equals(
            b[field].astype("string").fillna("<missing>")
        ), f"SAV/CSV mismatch for {field}"
    print(f"PASS: SAV/CSV agree respondent by respondent for all {len(a):,} IDs")


def sensitivity(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set ONLY gender to missing for 21 'Other' answers; never remove rows."""
    other = base.gender_label.eq("Other").fillna(False)
    alternative = base.copy(deep=True)
    alternative.loc[other, ["gender_clean", "gender_label"]] = pd.NA
    pd.testing.assert_frame_equal(base.loc[~other], alternative.loc[~other], check_exact=True)
    assert len(base) == len(alternative) == 28_333
    assert alternative.id.equals(base.id)
    assert int(other.sum()) == 21
    assert int(base.gender_label.isna().sum()) == 2
    assert int(alternative.gender_label.isna().sum()) == 23
    assert (base.gender_label.eq("Male").fillna(False) == alternative.gender_label.eq("Male").fillna(False)).all()
    assert (base.gender_label.eq("Female").fillna(False) == alternative.gender_label.eq("Female").fillna(False)).all()
    a = base.gender_label.value_counts(dropna=False)
    b = alternative.gender_label.value_counts(dropna=False)
    rows = []
    for group in ("Male", "Female", "Other", "Missing"):
        before = int(a.get(group, 0)) if group != "Missing" else int(base.gender_label.isna().sum())
        after = int(b.get(group, 0)) if group != "Missing" else int(alternative.gender_label.isna().sum())
        rows.append({"category": group, "keep_other": before, "set_other_missing": after, "difference": after-before})
    result = pd.DataFrame(rows)
    assert int(result.keep_other.sum()) == int(result.set_other_missing.sum()) == 28_333
    return alternative, result


def self_test() -> None:
    meta = SimpleNamespace(variable_value_labels={
        "country": {2: "Canada", 13: "Australia"},
        "gender": {1: "Male", 2: "Female"},
        "gender_australia": {1: "Male", 2: "Female", 3: "Other", 99: "Prefer not to say"},
    })
    raw = pd.DataFrame({
        "id": list(range(1, 8)),
        "country": [2, 2, 13, 13, 13, 13, 13],
        "gender": [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan],
        "gender_australia": [np.nan, np.nan, 1, 2, 3, np.nan, 99],
    })
    backup = raw.copy(deep=True)
    clean = clean_gender(raw, meta, strict=False)
    assert clean.gender_label.fillna("Missing").tolist() == [
        "Male", "Female", "Male", "Female", "Other", "Missing", "Missing"
    ]
    pd.testing.assert_frame_equal(raw, backup, check_exact=True)
    print("PASS: gender source routing, Other preserved, labelled refusal treated as missing")
    print("PASS: original source DataFrame unchanged; no row removed")
    wrong = raw.copy(deep=True)
    wrong.loc[0, "gender_australia"] = 3
    try:
        clean_gender(wrong, meta, strict=False)
    except AssertionError as err:
        assert "outside Australia" in str(err)
        print("PASS: Australian response in wrong country detected")
    else:
        raise AssertionError("Failed to detect country routing error")
    wrong = raw.copy(deep=True)
    wrong.loc[2, "gender_australia"] = 7
    try:
        clean_gender(wrong, meta, strict=False)
    except AssertionError as err:
        assert "Unexpected observed codes" in str(err)
        print("PASS: unlabelled gender code detected")
    else:
        raise AssertionError("Failed to detect unlabelled response code")
    wrong_meta = SimpleNamespace(variable_value_labels={
        **meta.variable_value_labels,
        "gender_australia": {1: "Male", 2: "Female", 3: "Don't know", 99: "Prefer not to say"},
    })
    try:
        clean_gender(raw, wrong_meta, strict=False)
    except AssertionError as err:
        assert "Unexpected Australian gender labels" in str(err)
        print("PASS: incorrectly describing Other as missing rejected")
    else:
        raise AssertionError("Failed to detect invalid gender metadata")
    print("TASK 17 SELF-TEST PASSED")


def main() -> None:
    import pyreadstat
    for path in (SAV, CSV):
        if not path.is_file():
            raise FileNotFoundError(f"Missing original source: {path}")

    print("=== TASK 17: MERGE GENDER FIELDS, PRESERVE 'OTHER' ===")
    original_sav, meta = pyreadstat.read_sav(
        str(SAV), apply_value_formats=False, user_missing=True
    )
    original_sav.columns = original_sav.columns.str.strip().str.lower()
    # Only the four relevant fields are needed from CSV for the independent cross-check.
    original_csv = pd.read_csv(
        CSV,
        usecols=lambda c: c.strip().lower() in {"id", "country", "gender", "gender_australia"},
        low_memory=False, encoding="utf-8-sig"
    )
    original_csv.columns = original_csv.columns.str.strip().str.lower()

    maps, labels = codebook_mapping(meta)
    print("Original SAV labels:")
    for field in ("gender", "gender_australia"):
        print(f"  {field}: {labels[field]}")
        print(f"  {field} recode mapping: {maps[field]}")

    # SAV raw copy, and a separate selection from original CSV.
    sav_before = original_sav.copy(deep=True)
    sav = clean_gender(original_sav, meta)
    csv = clean_gender(original_csv, meta)
    pd.testing.assert_frame_equal(original_sav, sav_before, check_exact=True)
    compare_source_and_output(sav, csv)

    # Pre-merge distribution: Australian blanks can differ from general country blanks.
    for field in ("gender", "gender_australia"):
        counts = numeric(original_sav[field]).value_counts(dropna=False)
        print(f"Original {field} counts: {counts.to_dict()}")

    print("\nUnified gender counts:")
    summary = pd.DataFrame([
        {"gender": k, "n": int(sav.gender_label.eq(k).fillna(False).sum())}
        for k in ("Male", "Female", "Other")
    ] + [{"gender": "Missing", "n": int(sav.gender_label.isna().sum())}])
    print(summary.to_string(index=False))
    assert summary.n.sum() == 28_333

    alt, sens = sensitivity(sav)
    print("\nSensitivity: set Australia's 21 'Other' answers to missing (do not drop rows):")
    print(sens.to_string(index=False))

    # Audit counts per country (metadata-derived names).
    countries = metadata_labels(meta, "country")
    group = (sav.assign(_country=numeric(sav.country).astype(int))
             .groupby("_country", sort=True))
    country_rows = []
    for code, g in group:
        country_rows.append({
            "country_code": code, "country": countries[code], "respondents": len(g),
            "male": int(g.gender_clean.eq(1).sum()),
            "female": int(g.gender_clean.eq(2).sum()),
            "other": int(g.gender_clean.eq(3).sum()),
            "missing_gender": int(g.gender_clean.isna().sum()),
        })
    country_audit = pd.DataFrame(country_rows)
    assert country_audit.loc[:, ["male", "female", "other", "missing_gender"]].sum().sum() == 28_333
    assert len(country_audit) == 24

    OUT.mkdir(exist_ok=True, parents=True)
    summary.to_csv(OUT / "task17_gender_summary.csv", index=False)
    country_audit.to_csv(OUT / "task17_gender_country_audit.csv", index=False)
    sens.to_csv(OUT / "task17_gender_sensitivity.csv", index=False)
    # No personally identifying respondent data in the decision or AI-use logs.
    pd.DataFrame([{
        "task": 17,
        "decision": "Merge gender and gender_australia by country; retain Australia's Other category",
        "alternative": "Set Australia's 21 Other responses to missing gender (keep respondents)",
        "reason": "Other is a valid SAV-labelled survey response, not a refusal",
        "effect_primary": "15,107 Male; 13,203 Female; 21 Other; 2 missing; 0 rows removed",
        "effect_alternative": "15,107 Male; 13,203 Female; 0 Other; 23 missing; 0 rows removed",
        "verification": "SAV value labels, original CSV cross-check by ID, immutable raw DataFrame",
        "status": "Verified by script; independent human run pending",
    }]).to_csv(OUT / "task17_decision_log_entry.csv", index=False)
    pd.DataFrame([{
        "task": 17,
        "ai_contribution": "Generated and self-tested labelled gender merge and alternative treatment",
        "independent_reference": "Original uploaded Pew CSV counts",
        "human_verification": "Run against original SAV and review terminal output",
        "ai_errors": "None found in Task 17 tests; report any discrepancies before sign-off",
    }]).to_csv(OUT / "task17_ai_use_log_entry.csv", index=False)
    (OUT / "task17_validation_note.txt").write_text(
        "Task 17 - merge gender and gender_australia. SAV labels determine response meaning.\n"
        "Keep 21 Australian 'Other' answers in the cleaned data.\n"
        "Alternative is to set ONLY their gender to missing (do not drop 21 respondents).\n"
        "Male vs Female comparison membership is identical under both decisions.\n"
        "Unweighted distribution: Male=15107, Female=13203, Other=21, Missing=2.\n"
        "Source data and existing shared pew_clean.csv were not modified.\n",
        encoding="utf-8",
    )
    print("\nPASS: raw data unchanged; all 28,333 rows retained")
    print("PASS: 21 Other kept; 2 original missing genders preserved")
    print("PASS: SAV/CSV agree person by person for raw and cleaned gender")
    print("Saved: task17_gender_summary.csv, task17_gender_country_audit.csv,")
    print("       task17_gender_sensitivity.csv, task17_decision_log_entry.csv,")
    print("       task17_ai_use_log_entry.csv, task17_validation_note.txt")
    print("TASK 17 PASSED: gender merged; Other preserved; alternative tested without dropping respondents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
