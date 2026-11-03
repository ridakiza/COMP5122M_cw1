"""COMP5122M P1 — Task 11: clean Q32 regulation responses without losing 'Not sure'.

From the repository root:
    python scripts/validate_task11.py --self-test
    python scripts/validate_task11.py

We identify nonresponse from each original SAV value label, not from the code
number. In Q32, 8 = 'Not sure' (VALID), whereas 9 = 'Refused' (missing).
The reference counts are assertions against the independently inspected
original CSV; they are NOT used to decide which values to recode.

No source files or pew_clean.csv are overwritten. Only audit outputs are saved.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from validate_task04 import is_nonresponse, canonical_label

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "clean"
SAV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset.sav"
CSV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv"
QUESTIONS = ["aireg_us", "aireg_china", "aireg_eu", "aireg_country"]
COLS = ["id", "country", *QUESTIONS]
POSSIBLE_CODES = {1, 2, 3, 4, 8, 9}
EXPECTED_N = 28_333
EXPECTED_COUNTRIES = 24

# Verified against uploaded ORIGINAL CSV (before cleaning), not recoding rules.
REFERENCE_COUNTS = {
    "aireg_us":      {1: 4105, 2: 8501, 3: 6559, 4: 5694, 8: 3112, 9: 362},
    "aireg_china":   {1: 3367, 2: 5628, 3: 6592, 4: 8990, 8: 3376, 9: 380},
    "aireg_eu":      {1: 4013, 2: 10809, 3: 5566, 4: 3345, 8: 4019, 9: 581},
    "aireg_country": {1: 6758, 2: 9973, 3: 5711, 4: 2732, 8: 2946, 9: 213},
}


def metadata_labels(meta: object, var: str) -> dict[int, str]:
    """Get numeric code labels for one question directly from original SAV."""
    mapping = getattr(meta, "variable_value_labels", None) or {}
    match = [k for k in mapping if str(k).strip().lower() == var]
    assert len(match) == 1, f"{var}: expected exactly one SAV label map, got {match}"
    result = {}
    for raw_code, text in mapping[match[0]].items():
        number = float(raw_code)
        assert number.is_integer(), f"{var}: non-integer labelled code {raw_code!r}"
        code = int(number)
        assert code not in result, f"{var}: duplicate SAV label for code {code}"
        result[code] = str(text)
    return result


def missing_from_labels(meta: object, var: str) -> set[int]:
    """Determine missing codes from response *descriptions*; verify semantics."""
    labels = metadata_labels(meta, var)
    assert set(labels) == POSSIBLE_CODES, (
        f"{var}: unexpected value-label codes {sorted(labels)}. Stop and review."
    )
    missing = {code for code, text in labels.items() if is_nonresponse(text)}
    assert canonical_label(labels[8]) == "not sure", (
        f"{var}: expected code 8 to mean 'Not sure', got {labels[8]!r}"
    )
    assert "refus" in canonical_label(labels[9]), (
        f"{var}: expected code 9 to mean refusal, got {labels[9]!r}"
    )
    assert missing == {9}, (
        f"{var}: label interpretation yielded {sorted(missing)} rather than [9]"
    )
    return missing


def response_counts(series: pd.Series, var: str) -> dict[int, int]:
    """Count every observed original code, including 8/9, without recoding."""
    x = pd.to_numeric(series, errors="raise")
    assert x.notna().all(), f"{var}: unexpected blank original Q32 response"
    assert np.isfinite(x.to_numpy(dtype=float)).all(), f"{var}: nonfinite response"
    invalid = set(x.unique()) - POSSIBLE_CODES
    assert not invalid, f"{var}: unexpected values {invalid}"
    return {code: int(x.eq(code).sum()) for code in sorted(POSSIBLE_CODES)}


def check_frame(df: pd.DataFrame, name: str, n: int, countries: int | None) -> None:
    assert set(COLS).issubset(df.columns), f"{name}: fields missing {set(COLS) - set(df.columns)}"
    assert len(df) == n, f"{name}: expected {n} rows, found {len(df)}"
    assert df.id.notna().all() and df.id.is_unique, f"{name}: missing or repeated respondent IDs"
    assert df.country.notna().all(), f"{name}: missing country"
    if countries is not None:
        assert df.country.nunique() == countries, f"{name}: wrong country count"
    for var in QUESTIONS:
        response_counts(df[var], f"{name}.{var}")


def clean_q32(df: pd.DataFrame, meta: object, *, reference: bool = True,
              show: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (new cleaned DataFrame, audit) without modifying original."""
    check_frame(df, "input", EXPECTED_N if reference else len(df),
                EXPECTED_COUNTRIES if reference else None)
    original_snapshot = df.copy(deep=True)
    clean = df.copy(deep=True)
    audit = []
    for var in QUESTIONS:
        codes = missing_from_labels(meta, var)
        labels = metadata_labels(meta, var)
        counts = response_counts(df[var], var)
        if reference:
            assert counts == REFERENCE_COUNTS[var], (
                f"{var}: original responses differ from independent CSV reference: "
                f"{counts} vs {REFERENCE_COUNTS[var]}"
            )

        x = pd.to_numeric(df[var], errors="raise")
        mask = x.isin(codes)
        clean[var] = x.mask(mask, np.nan)
        changed = int(mask.sum())
        n_valid = int(clean[var].notna().sum())
        n_missing = int(clean[var].isna().sum())
        assert changed == counts[9] == n_missing, f"{var}: refusal count mismatch"
        assert n_valid == len(df) - changed, f"{var}: invalid valid-response count"
        assert clean.loc[x.eq(8), var].eq(8).all(), f"{var}: 'Not sure' was lost"
        assert int(clean[var].eq(8).sum()) == counts[8], f"{var}: code 8 count changed"
        for code in (1, 2, 3, 4):
            assert clean.loc[x.eq(code), var].eq(code).all(), f"{var}: code {code} changed"
        assert clean.loc[x.eq(9), var].isna().all(), f"{var}: refusals not recoded"
        if show:
            print(f"\n{var}")
            print("  Original SAV value labels:", labels)
            print("  Label-inferred missing codes:", sorted(codes))
            print("  Before counts:", counts)
            print(f"  Code 8 'Not sure' preserved: {counts[8]:,}")
            print(f"  Code 9 'Refused' recoded to NaN: {changed:,}")
            print(f"  After: {n_valid:,} valid, {n_missing:,} missing")
        audit.append({
            "variable": var,
            "source_n": len(df),
            "nonresponse_codes_from_sav_labels": ",".join(map(str, sorted(codes))),
            **{f"original_code_{code}": counts[code] for code in sorted(POSSIBLE_CODES)},
            "not_sure_preserved": counts[8],
            "refused_converted_to_nan": changed,
            "valid_after": n_valid,
            "missing_after": n_missing,
            "rows_removed": 0,
        })

    # Verify that no other fields, values, IDs, row order, or original cells changed.
    pd.testing.assert_frame_equal(df, original_snapshot, check_exact=True)
    pd.testing.assert_frame_equal(clean[["id", "country"]], df[["id", "country"]], check_exact=True)
    non_q32 = [c for c in df if c not in QUESTIONS]
    pd.testing.assert_frame_equal(clean[non_q32], df[non_q32], check_exact=True)
    assert clean is not df and clean.index.equals(df.index) and len(clean) == len(df)
    return clean, pd.DataFrame(audit)


def compare_by_id(sav: pd.DataFrame, csv: pd.DataFrame, stage: str) -> None:
    """Strict SAV/CSV comparison independent of row order, including NaNs."""
    a = sav[COLS].set_index("id").sort_index()
    b = csv[COLS].set_index("id").sort_index()
    assert a.index.equals(b.index), f"{stage}: SAV/CSV respondent IDs differ"
    for var in COLS[1:]:
        x = pd.to_numeric(a[var], errors="raise").to_numpy(dtype=float)
        y = pd.to_numeric(b[var], errors="raise").to_numpy(dtype=float)
        matches = np.isclose(x, y, rtol=0, atol=0, equal_nan=True)
        assert matches.all(), f"{stage}: SAV/CSV differ for {var} in {(~matches).sum()} respondents"
    print(f"PASS: SAV/CSV agree for all {len(a):,} respondents {stage}")


def make_country_audit(sav: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in QUESTIONS:
        x = pd.DataFrame({
            "country_code": pd.to_numeric(sav.country, errors="raise").astype(int),
            "code8_kept": pd.to_numeric(sav[var], errors="raise").eq(8),
            "code9_to_nan": pd.to_numeric(sav[var], errors="raise").eq(9),
            "valid_after": clean[var].notna(),
        })
        result = x.groupby("country_code", sort=True).agg(
            respondents=("valid_after", "size"),
            n_not_sure_preserved=("code8_kept", "sum"),
            n_refused_recoded=("code9_to_nan", "sum"),
            n_valid_after=("valid_after", "sum"),
        ).reset_index()
        result.insert(0, "variable", var)
        rows.append(result)
    combined = pd.concat(rows, ignore_index=True)
    assert len(combined) == 4 * EXPECTED_COUNTRIES
    return combined


def self_test() -> None:
    label_map = {1: "A lot of trust", 2: "Some trust", 3: "Not too much trust",
                 4: "No trust at all", 8: "Not sure", 9: "Refused (DO NOT READ)"}
    meta = SimpleNamespace(variable_value_labels={var: label_map for var in QUESTIONS})
    fake = pd.DataFrame({
        "id": [101, 102, 103, 104, 105, 106], "country": [2] * 6,
        **{var: [1, 2, 3, 4, 8, 9] for var in QUESTIONS},
    })
    before = fake.copy(deep=True)
    clean, audit = clean_q32(fake, meta, reference=False, show=False)
    assert all(audit.refused_converted_to_nan == 1)
    for var in QUESTIONS:
        assert clean.loc[4, var] == 8, f"{var}: lost valid 'Not sure'"
        assert pd.isna(clean.loc[5, var]), f"{var}: did not remove refusal"
    pd.testing.assert_frame_equal(fake, before, check_exact=True)
    print("PASS: all four Q32 items preserve 'Not sure', recode refusals and leave raw data unchanged")

    corrupt_labels = {**label_map, 8: "Don't know"}
    broken_meta = SimpleNamespace(variable_value_labels={var: corrupt_labels for var in QUESTIONS})
    try:
        clean_q32(fake, broken_meta, reference=False, show=False)
    except AssertionError:
        print("PASS: erroneous treatment of code 8 as missing rejected")
    else:
        raise AssertionError("Self-test failed: code 8 mislabelling not detected")

    bad_data = fake.copy()
    bad_data.loc[0, QUESTIONS[0]] = 7
    try:
        clean_q32(bad_data, meta, reference=False, show=False)
    except AssertionError:
        print("PASS: unknown response code detected")
    else:
        raise AssertionError("Self-test failed: unknown response was accepted")

    bad_csv = fake.copy()
    bad_csv.loc[0, QUESTIONS[0]] = 2
    try:
        compare_by_id(fake, bad_csv, "self-test")
    except AssertionError:
        print("PASS: respondent-level SAV/CSV discrepancy detected")
    else:
        raise AssertionError("Self-test failed: mismatched answers were accepted")
    print("TASK 11 SELF-TEST PASSED")


def main() -> None:
    print("=== TASK 11: Q32 AI REGULATION — KEEP 8, RECODE ONLY REFUSALS ===")
    for path in (SAV, CSV):
        if not path.is_file():
            raise FileNotFoundError(f"Missing original survey file: {path}")
    try:
        import pyreadstat
    except ImportError as exc:
        raise ImportError("Install pyreadstat: python -m pip install pyreadstat") from exc

    # Read only needed original fields; never replace response codes while loading.
    sav, meta = pyreadstat.read_sav(str(SAV), usecols=COLS,
                                   apply_value_formats=False, user_missing=True)
    csv = pd.read_csv(CSV, usecols=COLS, low_memory=False, encoding="utf-8-sig")
    sav.columns = sav.columns.str.strip().str.lower()
    csv.columns = csv.columns.str.strip().str.lower()
    check_frame(sav, "SAV", EXPECTED_N, EXPECTED_COUNTRIES)
    check_frame(csv, "CSV", EXPECTED_N, EXPECTED_COUNTRIES)
    compare_by_id(sav, csv, "before cleaning")

    clean_sav, audit_sav = clean_q32(sav, meta)
    clean_csv, audit_csv = clean_q32(csv, meta, show=False)
    pd.testing.assert_frame_equal(audit_sav, audit_csv, check_exact=True)
    compare_by_id(clean_sav, clean_csv, "after cleaning")

    print("\nTotals:")
    total_refusals = int(audit_sav.refused_converted_to_nan.sum())
    total_not_sure = int(audit_sav.not_sure_preserved.sum())
    print(f"  Refusals recoded across four question-items: {total_refusals:,}")
    print(f"  Valid 'Not sure' responses preserved across four items: {total_not_sure:,}")
    assert total_refusals == 1536 and total_not_sure == 13453
    assert audit_sav.source_n.eq(EXPECTED_N).all()
    assert audit_sav.rows_removed.eq(0).all()

    OUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUT / "task11_q32_audit.csv"
    countries_path = OUT / "task11_q32_country_audit.csv"
    note_path = OUT / "task11_validation_note.txt"
    audit_sav.to_csv(audit_path, index=False)
    make_country_audit(sav, clean_sav).to_csv(countries_path, index=False)
    note_path.write_text(
        "TASK 11 — Q32 AI regulation missing-response handling\n"
        "DECISION: Read each Q32 variable's original SAV labels. Recode only labelled "
        "'Refused' responses (code 9) to NaN. Preserve 'Not sure' (code 8) and "
        "substantive responses 1-4.\n"
        "ALTERNATIVE: Replace both codes 8 and 9 with missing as for Q30/Q31. "
        "Rejected: Q32 code 8 is an explicitly valid 'Not sure' category.\n"
        "EFFECT: Refusals recoded: US 362, China 380, EU 581, own country 213 "
        "(1536 total question-responses). 'Not sure' retained: US 3112, China "
        "3376, EU 4019, own country 2946 (13453 total question-responses). "
        "No respondents or other fields removed or modified.\n"
        "AI USE: ChatGPT drafted validation and cleaning code, independently checked "
        "original CSV counts and synthetic error detection. Run against SAV on the "
        "user's computer for second verification.\n"
        "AI ERRORS: Record separately if the SAV run or a code review reveals any.\n"
        "NOTE: This standalone check does not update data/clean/pew_clean.csv; "
        "incorporate the verified transformation in src/cleaning.py.\n",
        encoding="utf-8",
    )
    print("\nSaved:", audit_path.name, countries_path.name, note_path.name)
    print("TASK 11 PASSED: all four Q32 fields retain valid code 8, recode only 9,")
    print("                preserve 28,333 respondents and match SAV/CSV.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
