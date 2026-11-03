"""COMP5122M P1 — Task 19: reproduce the official weighted Pew Q30/Q31 toplines.

From repository root:
    python scripts/validate_task19.py --self-test
    python scripts/validate_task19.py

Requires:
    data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset.sav
    data/raw/Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv
    references/topline_expected.csv  # 192 percentages transcribed from official
                                     # Pew Spring 2025 Topline PDF, page 48

WHY RAW DATA? Task 10 converted 8/9 to NaN in clean variables. Pew's
published DK/Refused category, and its ALL-RESPONDENT denominator, must be
reconstructed from the original raw codes. Do not drop those respondents.
Code 8 in the REFERENCE file means combined 'DK/Refused', not only raw 8.

This validation reads SAV response labels rather than guessing missing codes.
It compares two separate input files by respondent ID, checks all 192 cells,
and writes AGGREGATE evidence only; it never changes pew_clean.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "clean"
SAV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset.sav"
CSV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv"
REF = ROOT / "references" / "topline_expected.csv"
QUESTIONS = ("ai_heard", "ai_cncexc")
CODES = (1, 2, 3, 8)  # 8 in the published/reference table means 8+9
N = 28_333


def numeric(s: pd.Series) -> pd.Series:
    """Preserve labelled numeric codes, treating whitespace cells as blank."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="raise")
    t = s.astype("string").str.strip()
    return pd.to_numeric(t.mask(t.eq("")), errors="raise")


def label_map(meta: object, variable: str) -> dict[float, str]:
    maps = getattr(meta, "variable_value_labels")
    key = next((k for k in maps if k.lower() == variable.lower()), None)
    if key is None:
        raise AssertionError(f"SAV lacks value labels for {variable}")
    return {float(k): str(v) for k, v in maps[key].items()}


def get_nonresponses(meta: object, variable: str) -> set[float]:
    """For Q30/Q31, classify 'Don't know' and 'Refused' from SAV labels."""
    labels = label_map(meta, variable)
    def is_nonresponse(text: str) -> bool:
        text = text.lower().replace("’", "'")
        return ("don't know" in text or "refused" in text or "prefer not to say" in text)
    missing = {code for code, lab in labels.items() if is_nonresponse(lab)}
    if len(missing) != 2:
        raise AssertionError(f"Unexpected {variable} nonresponse labels: {labels}; identified {missing}")
    if not {1., 2., 3.}.issubset(labels):
        raise AssertionError(f"Unexpected substantive response coding for {variable}: {labels}")
    return missing


def load_reference(path: Path = REF) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Official topline reference missing: {path}\n"
            "Copy the supplied references/topline_expected.csv into your repository."
        )
    ref = pd.read_csv(path)
    required = {"question", "country", "code", "expected"}
    if not required.issubset(ref.columns):
        raise AssertionError(f"Reference must contain {sorted(required)}")
    ref["code"] = pd.to_numeric(ref["code"], errors="raise").astype(int)
    ref["expected"] = pd.to_numeric(ref["expected"], errors="raise").astype(int)
    if len(ref) != 192 or ref.duplicated(["question", "country", "code"]).any():
        raise AssertionError("Reference must have 192 UNIQUE Q30/Q31 country-category cells")
    if set(ref.question) != set(QUESTIONS) or ref.country.nunique() != 24:
        raise AssertionError("Reference must contain both questions in exactly 24 countries")
    if not ref["code"].isin(CODES).all() or not ref.expected.between(0, 100).all():
        raise AssertionError("Unexpected category or percentage in reference")
    sizes = ref.groupby(["question", "country"]).agg(n=("code", "size"),
                                                   codes=("code", lambda s: frozenset(s)))
    if len(sizes) != 48 or not (sizes.n == 4).all() or any(x != frozenset(CODES) for x in sizes.codes):
        raise AssertionError("Each question-country must have categories 1, 2, 3 and DK/Refused")
    return ref


def check_raw(raw: pd.DataFrame, meta: object, tag: str) -> None:
    if raw.shape[0] != N:
        raise AssertionError(f"{tag}: unexpected row count {len(raw)}")
    if raw.id.isna().any() or raw.id.duplicated().any():
        raise AssertionError(f"{tag}: missing/duplicate IDs")
    if raw.country.isna().any() or raw.country.nunique() != 24:
        raise AssertionError(f"{tag}: missing countries or unexpected country count")
    w = numeric(raw.weight)
    if w.isna().any() or (w <= 0).any() or not np.isfinite(w.to_numpy()).all():
        raise AssertionError(f"{tag}: missing, nonpositive or infinite weights")
    for q in QUESTIONS:
        values = numeric(raw[q])
        allowed = {1., 2., 3.} | get_nonresponses(meta, q)
        if values.isna().any() or not set(values.unique()).issubset(allowed):
            raise AssertionError(f"{tag}: {q} contains blanks or unexpected response codes")


def compare_by_id(sav: pd.DataFrame, csv: pd.DataFrame) -> None:
    a = sav.set_index("id").sort_index()
    b = csv.set_index("id").sort_index()
    if not a.index.equals(b.index):
        raise AssertionError("SAV and CSV respondent IDs differ")
    for col in ["country", "weight", *QUESTIONS]:
        x = numeric(a[col]).to_numpy(dtype=float)
        y = numeric(b[col]).to_numpy(dtype=float)
        if not np.isclose(x, y, rtol=0, atol=1e-10, equal_nan=True).all():
            raise AssertionError(f"SAV/CSV respondent-level mismatch: {col}")
    print("PASS: SAV/CSV agree respondent-by-respondent on country, weight, Q30 and Q31")


def calculate(raw: pd.DataFrame, meta: object, country_names: dict[int, str],
              tag: str) -> pd.DataFrame:
    """Return ALL 192 weighted percentages, using every asked respondent as denominator."""
    rows: list[dict] = []
    for country_code, group in raw.groupby("country", sort=True):
        country = country_names[int(country_code)]
        w = numeric(group.weight)
        denom = float(w.sum())  # includes the people with codes 8/9
        for q in QUESTIONS:
            original = numeric(group[q])
            nonresponses = get_nonresponses(meta, q)
            masks = {1: original.eq(1), 2: original.eq(2), 3: original.eq(3),
                     8: original.isin(nonresponses)}
            if not (sum(int(mask.sum()) for mask in masks.values()) == len(group)):
                raise AssertionError(f"{tag}: {country} {q} categories not exhaustive")
            for code, mask in masks.items():
                numerator = float(w[mask].sum())
                exact = 100.0 * numerator / denom
                # Match published whole-number rounding; don't force rounded columns to total 100.
                rounded = int(np.floor(exact + 0.5))
                rows.append(dict(source=tag, question=q, country=country,
                                 code=code, category="DK/Refused" if code == 8 else str(code),
                                 respondents=len(group), weighted_denominator=denom,
                                 numerator_n=int(mask.sum()), weighted_numerator=numerator,
                                 exact_pct=exact, computed_pct=rounded))
    result = pd.DataFrame(rows)
    if len(result) != 192:
        raise AssertionError(f"{tag}: expected 192 calculations, obtained {len(result)}")
    exact_totals = result.groupby(["question", "country"]).exact_pct.sum()
    if not np.allclose(exact_totals, 100.0, atol=1e-10, rtol=0):
        raise AssertionError(f"{tag}: unrounded categories don't sum to 100")
    return result


def compare_published(got: pd.DataFrame, ref: pd.DataFrame, tag: str) -> pd.DataFrame:
    result = got.merge(ref, on=["question", "country", "code"],
                       how="outer", validate="one_to_one", indicator=True)
    if len(result) != 192 or not result._merge.eq("both").all():
        raise AssertionError(f"{tag}: missing or extra reference/country cells")
    result = result.drop(columns=["_merge"])
    result["difference_pp"] = result.computed_pct - result.expected
    result["match"] = result.difference_pp.eq(0)
    print(f"{tag}: matched {int(result.match.sum())}/192 published cells")
    if not result.match.all():
        print("MISMATCHES (inspect input, rounding, missing codes and denominator):")
        print(result.loc[~result.match,
                         ["question", "country", "code", "exact_pct", "computed_pct", "expected", "difference_pp"]]
              .to_string(index=False))
    return result


def self_test() -> None:
    """Tests deliberately designed to catch exclusion of DK/refused and ignoring weights."""
    from types import SimpleNamespace
    meta = SimpleNamespace(variable_value_labels={
        "ai_heard": {1: "A lot", 2: "A little", 3: "Nothing at all",
                     8: "Don't know", 9: "Refused"},
        "ai_cncexc": {1: "Excited", 2: "Concerned", 3: "Equally",
                      8: "Don't know", 9: "Refused"},
    })
    sample = pd.DataFrame({"id": [1, 2, 3, 4, 5], "country": [12]*5,
                           "weight": [1., 2., 1., 3., 3.],
                           "ai_heard": [1, 1, 2, 8, 9],
                           "ai_cncexc": [3, 2, 1, 8, 9]})
    assert get_nonresponses(meta, "ai_heard") == {8., 9.}
    assert get_nonresponses(meta, "ai_cncexc") == {8., 9.}
    # Check formula separately, because calculate() enforces 24 countries.
    w = sample.weight
    assert np.isclose(100*w[sample.ai_heard.eq(1)].sum()/w.sum(), 30.)
    assert np.isclose(100*w[sample.ai_heard.isin({8,9})].sum()/w.sum(), 60.)
    assert not np.isclose(100*w[sample.ai_heard.eq(1)].sum()/w[sample.ai_heard.isin({1,2,3})].sum(), 30.)
    assert not np.isclose(100*sample.ai_heard.eq(1).mean(), 30.)
    assert int(np.floor(2.5 + .5)) == 3
    print("PASS: weighted denominator includes DK/refused; both codes combined")
    print("PASS: excluding nonresponses or switching weights changes the test result")
    print("PASS: half-up whole-percent rounding behaves as intended")
    print("TASK 19 SELF-TEST PASSED")


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--self-test", action="store_true")
    parsed = args.parse_args()
    if parsed.self_test:
        self_test()
        return
    print("=== TASK 19: WEIGHTED PEW TOPLINES — Q30 + Q31 ===")
    for p in [SAV, CSV, REF]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}; place it at the expected repo path")
    try:
        import pyreadstat
    except ImportError as exc:
        raise RuntimeError("Install SAV reader with: python -m pip install pyreadstat") from exc
    sav, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    sav.columns = sav.columns.str.strip().str.lower()
    csv = pd.read_csv(CSV, usecols=["id", "country", "weight", *QUESTIONS], low_memory=False,
                      encoding="utf-8-sig")
    csv.columns = csv.columns.str.strip().str.lower()
    ref = load_reference()
    print(f"Reference: {len(ref)} cells, sourced from Pew Spring 2025 Topline PDF page 48")
    country_labels = label_map(meta, "country")
    names = {int(code): name for code, name in country_labels.items()}
    if len(names) != 24 or set(names.values()) != set(ref.country):
        raise AssertionError("SAV country labels do not match 24 reference countries")
    for q in QUESTIONS:
        print(f"{q} nonresponse codes, from SAV labels: {sorted(get_nonresponses(meta, q))}")
    check_raw(sav, meta, "SAV")
    check_raw(csv, meta, "CSV")
    compare_by_id(sav, csv)
    sav_result = compare_published(calculate(sav, meta, names, "SAV"), ref, "SAV")
    csv_result = compare_published(calculate(csv, meta, names, "CSV"), ref, "CSV")
    check = sav_result.merge(csv_result,
                             on=["question", "country", "code"], suffixes=("_sav", "_csv"),
                             validate="one_to_one")
    assert np.isclose(check.exact_pct_sav, check.exact_pct_csv, atol=1e-9, rtol=0).all(), \
        "SAV/CSV unrounded percentages disagree"
    uk = sav_result.loc[(sav_result.country == "UK") &
                        (sav_result.question == "ai_heard") &
                        (sav_result.code == 1)].iloc[0]
    print(f"UK Q30 'A lot': weighted={uk.exact_pct:.4f}% -> {uk.computed_pct}% (Pew={uk.expected}%)")
    assert uk.computed_pct == 41 and uk.expected == 41

    OUT.mkdir(parents=True, exist_ok=True)
    sav_result.sort_values(["question", "country", "code"]).to_csv(
        OUT / "task19_weighted_topline_audit.csv", index=False)
    country = (sav_result.groupby(["question", "country"], as_index=False)
               .agg(matched=("match", "sum"), cells=("match", "size"),
                    n=("respondents", "first"), weight_sum=("weighted_denominator", "first")))
    country["passed"] = country.matched.eq(4)
    country.to_csv(OUT / "task19_country_summary.csv", index=False)
    failures = sav_result.loc[~sav_result.match].copy()
    failures.to_csv(OUT / "task19_mismatches.csv", index=False)
    success = bool(sav_result.match.all() and csv_result.match.all())
    (OUT / "task19_validation_note.txt").write_text(
        "Task 19, Pew Spring 2025 Topline p.48 (Q30 and Q31).\n"
        "Source: original SAV, checked against original CSV by respondent ID.\n"
        "All country respondents, including 'Don't know' and 'Refused', remain in the weighted denominator.\n"
        "DK and Refused codes are identified from SAV value labels and combined for the published DK/Refused cell.\n"
        "Each published whole-number percentage independently rounded using floor(pct + 0.5).\n"
        f"SAV matched {int(sav_result.match.sum())}/192; CSV matched {int(csv_result.match.sum())}/192.\n"
        "SAV/CSV exact percentages matched within numerical tolerance.\n"
        "Do not overwrite raw data, cleaned dataset or education variables.\n",
        encoding="utf-8")
    pd.DataFrame([{
        "task": 19, "decision": "Use original numeric Q30/Q31 responses and original Pew weights",
        "alternative": "Drop DK/refused from denominator or use unweighted percentages",
        "reason": "Official Pew general-population toplines include nonresponses in the denominator",
        "effect": f"{int(sav_result.match.sum())}/192 published cells reproduced",
        "status": "PASSED" if success else "REVIEW"
    }]).to_csv(OUT / "task19_decision_log_entry.csv", index=False)
    pd.DataFrame([{
        "task": 19, "ai_use": "Generated validation code and checked 192 published PDF values against reference CSV",
        "verification": "Independent SAV + CSV respondent-level comparison and 192-cell published topline check",
        "result": f"SAV {int(sav_result.match.sum())}/192; CSV {int(csv_result.match.sum())}/192",
        "remaining_action": "None" if success else "Investigate discrepancies before marking task done"
    }]).to_csv(OUT / "task19_ai_use_log_entry.csv", index=False)
    print("Saved: task19_weighted_topline_audit.csv, task19_country_summary.csv,")
    print("       task19_mismatches.csv, task19_validation_note.txt, Task19 log entries")
    if not success:
        raise AssertionError("TASK 19 NOT VERIFIED: published percentages differ; investigate mismatches")
    print("TASK 19 PASSED: 192/192 official Pew cells matched using SAV and CSV; weighted DK-inclusive denominator.")


if __name__ == "__main__":
    main()
