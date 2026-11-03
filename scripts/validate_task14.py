"""COMP5122M P1 — Task 14: defensible age-band recovery.

From the repository ROOT:
    python scripts/validate_task14.py --self-test
    python scripts/validate_task14.py

The age follow-up bands vary by COUNTRY. Recover an age *band* only when
its entire interval fits in exactly one of 18–29, 30–49, 50–64 and 65+.
Do not infer exact ages, use interval midpoints or discard respondents.

Task 14 concerns non-Australian respondents with age=98 or 99.
Australia's separate age_australia coding is reserved for Task 16.
The Israeli follow-up on two respondents with exact ages is logged for Task 15.

Data remains local under data/clean/; do not commit respondent-level outputs.
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
BANDS = [("18-29", 18, 29), ("30-49", 30, 49), ("50-64", 50, 64), ("65+", 65, float("inf"))]
COUNTRY_SUFFIX_OVERRIDES = {"south africa": "safrica", "south korea": "skorea"}
REFERENCE = {"n": 28333, "missing_exact": 294, "recovered": 111,
             "still_missing_non_australia": 183, "followups_with_exact_age": 2,
             "recovered_by_band": {"18-29": 15, "30-49": 59, "50-64": 6, "65+": 31},
             "australia_age_present": 1827}


def metadata_labels(meta: object, field: str) -> dict[int, str]:
    """Read code=>meaning from the SAV, rather than hard-coding per-country bins."""
    all_labels = getattr(meta, "variable_value_labels", {}) or {}
    matches = [k for k in all_labels if str(k).strip().casefold() == field.casefold()]
    assert len(matches) == 1, f"Missing/ambiguous SAV labels for {field}: {matches}"
    result = {}
    for code, label in all_labels[matches[0]].items():
        n = float(code)
        assert n.is_integer(), f"Noninteger response code {code} in {field}"
        result[int(n)] = str(label).strip()
    return result


def normal_label(value: str) -> str:
    return str(value).strip().casefold().replace("’", "'").replace("‘", "'")


def interval_of_label(label: str) -> tuple[int, float] | None:
    """Parse country-age intervals, e.g. '30-44', '65+'; None = nonresponse.

    Raises if an unexpected label appears: never silently guess its meaning.
    """
    s = normal_label(label).replace("–", "-").replace("—", "-")
    if "don't know" in s or "do not know" in s or "refused" in s or "prefer not to say" in s:
        return None
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        low, high = map(int, m.groups())
        assert 18 <= low <= high <= 120, f"Unexpected age range: {label!r}"
        return low, high
    m = re.fullmatch(r"(\d+)\s*\+", s)
    if m:
        low = int(m.group(1))
        assert 18 <= low <= 120, f"Unexpected open-ended age range: {label!r}"
        return low, float("inf")
    raise AssertionError(f"Unrecognised follow-up age label {label!r}; review SAV dictionary")


def band_for_interval(age_interval: tuple[int, float]) -> str | None:
    """A band is admissible only if the ENTIRE follow-up interval fits inside it."""
    low, high = age_interval
    matches = [name for name, bmin, bmax in BANDS if bmin <= low and high <= bmax]
    return matches[0] if len(matches) == 1 else None


def prepare(raw: pd.DataFrame, fields: list[str], source: str,
            strict: bool = True) -> pd.DataFrame:
    """Prepare a numeric comparison COPY, turning CSV whitespace into nulls."""
    cols = [str(c).strip().casefold() for c in raw.columns]
    assert len(cols) == len(set(cols)), f"{source}: duplicate normalised names"
    names = dict(zip(raw.columns, cols))
    missing = sorted(set(fields) - set(cols))
    assert not missing, f"{source}: columns missing: {missing}"
    df = raw.rename(columns=names)[fields].copy(deep=True)
    for col in fields:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            texts = df[col].astype("string").str.strip()
            df[col] = texts.mask(texts.eq(""))
        df[col] = pd.to_numeric(df[col], errors="raise")
        assert np.isfinite(df[col].dropna().to_numpy(dtype=float)).all(), f"{source}: non-finite {col}"
    assert df.id.is_unique and df.id.notna().all(), f"{source}: duplicate or missing IDs"
    if strict:
        assert len(df) == REFERENCE["n"], f"{source}: incorrect sample size"
    return df


def country_age_columns(meta: object) -> tuple[dict[int, str], dict[int, str]]:
    country_labels = metadata_labels(meta, "country")
    mapping = {}
    for code, country_name in country_labels.items():
        name = normal_label(country_name)
        if name == "australia":
            continue
        suffix = COUNTRY_SUFFIX_OVERRIDES.get(name, name.replace(" ", ""))
        col = "d_age_" + suffix
        metadata_labels(meta, col)  # Ensure field definitions exist in original SAV.
        mapping[code] = col
    assert len(mapping) == 23, f"Expected 23 non-Australian age fields, found {len(mapping)}"
    return country_labels, mapping


def recover(df: pd.DataFrame, meta: object, strict: bool = True):
    """Return (output, country audit, Israeli/other exact-age follow-up audit)."""
    original = df.copy(deep=True)
    countries, mapping = country_age_columns(meta)
    aus_codes = [k for k, name in countries.items() if normal_label(name) == "australia"]
    assert len(aus_codes) == 1
    aus = df.country.eq(aus_codes[0])
    assert df.loc[aus, "age"].isna().all(), "Age must be structurally blank in Australia"
    assert df.loc[~aus, "age"].notna().all(), "Unexpected original blank exact age outside Australia"
    assert df.loc[~aus, "age_australia"].isna().all(), "Australian age variable populated elsewhere"
    assert df.loc[~aus, "age"].between(18, 99).all(), "Impossible exact-age code"
    assert df.loc[~aus, "age"].mod(1).eq(0).all(), "Non-integer exact age"
    label_age = metadata_labels(meta, "age")
    assert "97" in normal_label(label_age[97]) and "older" in normal_label(label_age[97])
    assert "know" in normal_label(label_age[98]) and "refus" in normal_label(label_age[99])

    out = df[["id", "country", "age", "age_australia"]].copy(deep=True)
    out["age_clean"] = out.age.mask(out.age.isin([98, 99]), np.nan)
    out["age_band_partial"] = pd.Series(pd.NA, index=df.index, dtype="string")
    out["age_band_source"] = "unclassified"
    out["recovery_status"] = "no_followup"
    out["followup_field"] = pd.Series(pd.NA, index=df.index, dtype="string")
    out["followup_label"] = pd.Series(pd.NA, index=df.index, dtype="string")

    # Those with usable exact ages already have an unambiguous band; no recovery needed.
    for band, minimum, maximum in BANDS:
        mask = ~aus & out.age_clean.between(minimum, maximum)
        out.loc[mask, "age_band_partial"] = band
        out.loc[mask, "age_band_source"] = "exact_age"
        out.loc[mask, "recovery_status"] = "exact_age_available"

    out.loc[aus, "age_band_source"] = "australia_pending_task16"
    out.loc[aus, "recovery_status"] = "task16_australia"
    missing_exact = ~aus & df.age.isin([98, 99])
    conflict_rows = []

    for country_code, col in mapping.items():
        country_mask = df.country.eq(country_code)
        values = df[col].dropna()
        assert values.mod(1).eq(0).all(), f"{col}: noninteger follow-up codes"
        # Task 8: non-country fields must be structurally empty.
        assert df.loc[~country_mask, col].isna().all(), f"{col}: responses outside the owning country"
        known_labels = metadata_labels(meta, col)
        # Assert every ACTUAL response code is documented by SAV metadata.
        stray = set(values.astype(int)) - set(known_labels)
        assert not stray, f"{col}: unexpected codes absent from SAV labels: {stray}"

        eligible = country_mask & missing_exact
        for idx, code in df.loc[eligible, col].items():
            if pd.isna(code):
                continue
            label = known_labels[int(code)]
            out.at[idx, "followup_field"] = col
            out.at[idx, "followup_label"] = label
            interval = interval_of_label(label)
            if interval is None:
                out.at[idx, "recovery_status"] = "followup_nonresponse"
            else:
                band = band_for_interval(interval)
                if band is None:
                    out.at[idx, "recovery_status"] = "overlaps_target_bands"
                else:
                    out.at[idx, "age_band_partial"] = band
                    out.at[idx, "age_band_source"] = "recovered_followup"
                    out.at[idx, "recovery_status"] = "recovered"

        # Log violations of questionnaire routing, but NEVER overwrite known age.
        unexpected = country_mask & df.age.between(18, 97) & df[col].notna()
        for idx, code in df.loc[unexpected, col].items():
            interval = interval_of_label(known_labels[int(code)])
            original_age = float(df.at[idx, "age"])
            consistent = interval is not None and interval[0] <= original_age <= interval[1]
            conflict_rows.append({
                "id": df.at[idx, "id"], "country": countries[country_code],
                "exact_age": original_age, "followup_field": col,
                "followup_code": int(code), "followup_label": known_labels[int(code)],
                "consistent_with_exact_age": bool(consistent),
                "action": "retain_exact_age; review_in_task15",
            })

    assert int(missing_exact.sum()) == int(out.loc[missing_exact].shape[0])
    n_recovered = int(out.age_band_source.eq("recovered_followup").sum())
    n_remaining = int(out.loc[missing_exact, "age_band_partial"].isna().sum())
    assert n_recovered + n_remaining == int(missing_exact.sum())
    assert out.loc[~missing_exact & ~aus, "age_band_source"].eq("exact_age").all()
    assert out.loc[missing_exact, "age_clean"].isna().all(), "Recovery must not invent exact ages"
    assert int(out.age_band_source.eq("recovered_followup").sum()) == n_recovered
    pd.testing.assert_frame_equal(df, original, check_exact=True)
    assert len(out) == len(df) and out.id.equals(df.id)

    country_audit = []
    for code, name in countries.items():
        subset = out.country.eq(code)
        original_missing = subset & missing_exact
        country_audit.append({
            "country": name,
            "missing_exact_ages": int(original_missing.sum()),
            "followup_present": int(out.loc[original_missing, "followup_field"].notna().sum()),
            "recovered": int((original_missing & out.age_band_source.eq("recovered_followup")).sum()),
            "still_missing": int((original_missing & out.age_band_partial.isna()).sum()),
            "ambiguous_overlap": int((original_missing & out.recovery_status.eq("overlaps_target_bands")).sum()),
            "followup_nonresponse": int((original_missing & out.recovery_status.eq("followup_nonresponse")).sum()),
            "followup_blank": int((original_missing & out.recovery_status.eq("no_followup")).sum()),
        })
    audit = pd.DataFrame(country_audit)
    conflicts = pd.DataFrame(conflict_rows, columns=["id", "country", "exact_age",
        "followup_field", "followup_code", "followup_label", "consistent_with_exact_age", "action"])
    if strict:
        assert int(missing_exact.sum()) == REFERENCE["missing_exact"]
        assert n_recovered == REFERENCE["recovered"]
        assert n_remaining == REFERENCE["still_missing_non_australia"]
        assert len(conflicts) == REFERENCE["followups_with_exact_age"]
        assert int(df.loc[aus, "age_australia"].notna().sum()) == REFERENCE["australia_age_present"]
        actual_band = out.loc[out.age_band_source.eq("recovered_followup"), "age_band_partial"].value_counts().to_dict()
        assert actual_band == REFERENCE["recovered_by_band"], actual_band
    return out, audit, conflicts


def compare_by_id(sav: pd.DataFrame, csv: pd.DataFrame, columns: list[str], stage: str) -> None:
    a, b = sav.set_index("id").sort_index(), csv.set_index("id").sort_index()
    assert a.index.equals(b.index), f"{stage}: respondent IDs differ"
    for col in columns:
        if col in ("age_band_partial", "age_band_source", "recovery_status", "followup_field", "followup_label"):
            equal = a[col].fillna("<NA>").astype(str).eq(b[col].fillna("<NA>").astype(str))
        else:
            equal = np.isclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float),
                               atol=0, rtol=0, equal_nan=True)
        assert bool(np.all(equal)), f"{stage}: {col} differs across SAV/CSV"
    print(f"PASS: SAV/CSV agree for {len(a):,} respondents {stage}")


def self_test() -> None:
    assert band_for_interval(interval_of_label("30-44")) == "30-49"
    assert band_for_interval(interval_of_label("75+")) == "65+"
    assert band_for_interval(interval_of_label("45-59")) is None
    assert band_for_interval(interval_of_label("50+")) is None
    assert band_for_interval(interval_of_label("25-34")) is None
    assert interval_of_label("Refused (DO NOT READ)") is None
    try:
        interval_of_label("Special category unspecified")
        raise AssertionError("Unknown labels must be rejected")
    except AssertionError as err:
        assert "Unrecognised" in str(err)
    print("PASS: whole-interval mapping succeeds; overlapping and unknown labels not guessed")

    meta = SimpleNamespace(variable_value_labels={
        "country": {2: "Canada", 13: "Australia", 18: "Israel"},
        "age": {97: "97 or older", 98: "Don't know", 99: "Refused"},
        "d_age_canada": {1: "18-29", 2: "30-39", 3: "40-49", 4: "50+", 9: "Refused"},
        "d_age_israel": {1: "18-29", 2: "30-44", 3: "45-64", 4: "65+", 9: "Refused"},
    })
    # Country-age-column mapping function is fixed for 23 real countries.
    toy = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7],
        "country": [2, 2, 2, 2, 18, 18, 13],
        "age": [98, 99, 97, 22, 65, 98, np.nan],
        "age_australia": [np.nan]*6+[3],
        "d_age_canada": [2, 4, np.nan, np.nan, np.nan, np.nan, np.nan],
        "d_age_israel": [np.nan, np.nan, np.nan, np.nan, 3, 4, np.nan],
    })
    # Temporarily furnish the minimal two-country mapping for toy data.
    global country_age_columns
    old_mapper = country_age_columns
    def toy_mapper(m):
        return metadata_labels(m,"country"), {2:"d_age_canada",18:"d_age_israel"}
    country_age_columns = toy_mapper
    try:
        before = toy.copy(deep=True)
        result, audit, conflicts = recover(toy, meta, strict=False)
        assert result.set_index("id").loc[1, "age_band_partial"] == "30-49"
        assert pd.isna(result.set_index("id").loc[2, "age_band_partial"])
        assert result.set_index("id").loc[3, "age_band_partial"] == "65+"
        assert result.set_index("id").loc[6, "age_band_partial"] == "65+"
        assert result.set_index("id").loc[5, "age_band_partial"] == "65+"
        assert len(conflicts)==1 and not bool(conflicts.loc[0,"consistent_with_exact_age"])
        assert pd.isna(result.set_index("id").loc[6, "age_clean"])
        pd.testing.assert_frame_equal(toy,before,check_exact=True)
        print("PASS: recovered only valid follow-up bands; exact age, originals and Australian routing preserved")
        wrong=toy.copy(deep=True); wrong.loc[0,"d_age_canada"]=42
        try:
            recover(wrong,meta,strict=False)
            raise AssertionError("Unknown follow-up code not detected")
        except AssertionError as err:
            assert "unexpected codes" in str(err)
        print("PASS: unexpected follow-up codes detected")
    finally:
        country_age_columns = old_mapper
    print("TASK 14 SELF-TEST PASSED")


def main() -> None:
    import pyreadstat
    for p in (SAV, CSV):
        if not p.is_file():
            raise FileNotFoundError(f"Original Pew file missing: {p}")
    print("=== TASK 14: COUNTRY AGE FOLLOW-UP RECOVERY ===")
    sav_raw, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    csv_raw = pd.read_csv(CSV, low_memory=False, encoding="utf-8-sig")
    assert sav_raw.shape == (28333,483) and csv_raw.shape == (28333,483)
    country_labels, mapping = country_age_columns(meta)
    fields = ["id", "country", "age", "age_australia"] + sorted(mapping.values())
    sav = prepare(sav_raw, fields, "SAV")
    csv = prepare(csv_raw, fields, "CSV")
    compare_by_id(sav, csv, fields[1:], "before age recovery")
    sav_out, country_audit, conflicts = recover(sav, meta)
    csv_out, csv_audit, csv_conflicts = recover(csv, meta)
    compare_by_id(sav_out, csv_out, [c for c in sav_out if c != "id"], "after age recovery")
    pd.testing.assert_frame_equal(country_audit, csv_audit)
    pd.testing.assert_frame_equal(conflicts, csv_conflicts, check_dtype=False)

    not_aus = ~sav.country.eq(next(k for k,v in country_labels.items() if normal_label(v)=="australia"))
    n_missing = int((not_aus & sav.age.isin([98,99])).sum())
    n_recovered = int(sav_out.age_band_source.eq("recovered_followup").sum())
    n_remaining = n_missing - n_recovered
    print(f"Exact ages missing outside Australia: {n_missing}")
    print(f"Unambiguous age bands recovered:     {n_recovered}")
    print(f"Still missing outside Australia:   {n_remaining}")
    print("\nRecovered bands:")
    print(sav_out.loc[sav_out.age_band_source.eq("recovered_followup"),"age_band_partial"].value_counts().reindex([b[0] for b in BANDS],fill_value=0).to_string())
    print("\nCountry recovery audit (countries with missing exact ages):")
    print(country_audit.loc[country_audit.missing_exact_ages.gt(0)].to_string(index=False))
    print("\nUnexpected age follow-ups where exact ages exist — Task 15:")
    print(conflicts[["country","exact_age","followup_label","consistent_with_exact_age","action"]].to_string(index=False))
    print("Australia: 1,827 reported age bands, 2 blanks; handled in Task 16.")
    print("All 28,333 respondents preserved; no exact ages imputed.")

    OUT.mkdir(parents=True, exist_ok=True)
    sav_out.to_csv(OUT/"task14_age_recovery_preview.csv",index=False)
    country_audit.to_csv(OUT/"task14_recovery_country_audit.csv",index=False)
    conflicts.to_csv(OUT/"task14_followup_conflicts_task15.csv",index=False)
    pd.DataFrame([{
        "task":14,
        "decision":"Assign age band only when the complete country-specific follow-up interval fits one target band",
        "alternative":"Assign midpoint or nearest target group to overlapping intervals",
        "reason":"Unknown exact ages must not be invented; crossed target boundaries cannot be resolved",
        "effect":f"{n_recovered} age bands recovered; {n_remaining} outside Australia remain missing; 0 exact ages imputed",
        "respondents_removed":0,
        "verification":"SAV labels, separate CSV, person-level matching, country audit, synthetic tests",
        "status":"verified_pending_user_SAV_run",
    }]).to_csv(OUT/"task14_decision_log_entry.csv",index=False)
    pd.DataFrame([{
        "task":14,
        "ai_contribution":"Proposed whole-interval recovery and wrote/testing validation code",
        "independent_evidence":"Pew dictionary follow-up labels + uploaded original CSV",
        "human_verification":"Pending user SAV and CSV run",
        "verification_outcome":"pending_user_verification",
        "ai_errors_identified":"none in Task14 diagnostic check",
        "correction":"",
    }]).to_csv(OUT/"task14_ai_use_log_entry.csv",index=False)
    (OUT/"task14_validation_note.txt").write_text(
        f"Task 14: {n_missing} missing exact ages outside Australia; {n_recovered} unambiguous "
        f"age bands recovered, {n_remaining} still missing. Do not infer exact ages; "
        "Australia is deferred to Task 16. Two Israeli follow-ups in records with an exact age "
        "are reserved for Task 15. The original SAV and CSV are untouched.\n",
        encoding="utf-8")
    print("\nSaved: task14_age_recovery_preview.csv, task14_recovery_country_audit.csv,")
    print("       task14_followup_conflicts_task15.csv, task14_decision_log_entry.csv,")
    print("       task14_ai_use_log_entry.csv, task14_validation_note.txt")
    print("TASK 14 PASSED: 111 unambiguous follow-up bands recovered; 183 unclassified outside Australia.")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
