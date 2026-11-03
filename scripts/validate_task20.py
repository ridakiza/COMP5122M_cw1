"""COMP5122M P1 — Task 20: examine unweighted vs weighted Pew Q30/Q31.

From the repository root:
    python scripts/validate_task20.py --self-test
    python scripts/validate_task20.py

Requires both original SAV and original CSV under data/raw/ and the 192-cell
reference from Task 19 under references/topline_expected.csv.

Use original responses for BOTH estimates; all respondents, including DK and
refusals, stay in each denominator. Nonresponse codes are read from SAV labels.
This is a comparison, not a request to replace Pew's weights. It never writes
or overwrites pew_clean.csv and only creates aggregate evidence/logs.
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
REF = ROOT / "references" / "topline_expected.csv"
QUESTIONS = ("ai_heard", "ai_cncexc")
CATEGORY_CODES = (1, 2, 3, 8)  # 8 is the COMBINED DK/refused table category
EXPECTED_N = 28_333


def numeric(series: pd.Series) -> pd.Series:
    """Strict numeric conversion, allowing blank/whitespace to mean NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="raise")
    s = series.astype("string").str.strip()
    return pd.to_numeric(s.mask(s.eq("")), errors="raise")


def labels(meta: object, field: str) -> dict[float, str]:
    all_maps = getattr(meta, "variable_value_labels", {})
    key = next((k for k in all_maps if str(k).strip().lower() == field), None)
    if key is None:
        raise AssertionError(f"Missing SAV value labels for {field}")
    return {float(k): str(v) for k, v in all_maps[key].items()}


def nonresponse_codes(meta: object, field: str) -> set[float]:
    labs = labels(meta, field)
    codes = {c for c, lab in labs.items() if "don't know" in lab.lower().replace("’", "'")
             or "refused" in lab.lower()}
    if codes != {8.0, 9.0}:
        raise AssertionError(f"Unexpected Q30/Q31 nonresponse labels for {field}: {labs}")
    if not {1.0, 2.0, 3.0}.issubset(labs):
        raise AssertionError(f"Missing substantive response labels for {field}: {labs}")
    return codes


def reference(path: Path = REF) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Task 19 reference: {path}")
    ref = pd.read_csv(path)
    required = {"country", "question", "code", "expected"}
    if not required.issubset(ref):
        raise AssertionError(f"Reference missing columns: {sorted(required - set(ref))}")
    ref["code"] = pd.to_numeric(ref.code, errors="raise").astype(int)
    ref["expected"] = pd.to_numeric(ref.expected, errors="raise").astype(int)
    if (len(ref) != 192 or ref.duplicated(["country", "question", "code"]).any()
            or set(ref.question) != set(QUESTIONS) or ref.country.nunique() != 24):
        raise AssertionError("Expected 192 unique official Q30/Q31 country-category cells")
    groups = ref.groupby(["country", "question"]).code.agg(lambda s: frozenset(s))
    if len(groups) != 48 or any(g != frozenset(CATEGORY_CODES) for g in groups):
        raise AssertionError("Official reference is missing a question/country/category")
    if not ref.expected.between(0, 100).all():
        raise AssertionError("Invalid reference percentages")
    return ref


def check_input(df: pd.DataFrame, meta: object, name: str, n: int = EXPECTED_N,
                n_countries: int = 24) -> None:
    if len(df) != n or df.id.isna().any() or df.id.duplicated().any():
        raise AssertionError(f"{name}: invalid respondent count or ID uniqueness")
    if df.country.isna().any() or df.country.nunique() != n_countries:
        raise AssertionError(f"{name}: missing country or unexpected country count")
    w = numeric(df.weight)
    if w.isna().any() or (w <= 0).any() or not np.isfinite(w).all():
        raise AssertionError(f"{name}: invalid survey weights")
    for q in QUESTIONS:
        vals = numeric(df[q])
        if vals.isna().any() or not set(vals.unique()).issubset({1., 2., 3.} | nonresponse_codes(meta, q)):
            raise AssertionError(f"{name}: unexpected or missing raw {q} response codes")


def compare_inputs(sav: pd.DataFrame, csv: pd.DataFrame) -> None:
    a = sav.set_index("id").sort_index()
    b = csv.set_index("id").sort_index()
    if not a.index.equals(b.index):
        raise AssertionError("SAV/CSV respondent IDs differ")
    for col in ["country", "weight", *QUESTIONS]:
        x = numeric(a[col]).to_numpy(dtype=float)
        y = numeric(b[col]).to_numpy(dtype=float)
        if not np.isclose(x, y, rtol=0, atol=1e-10, equal_nan=True).all():
            raise AssertionError(f"SAV/CSV respondent-level mismatch: {col}")
    print(f"PASS: SAV and CSV match respondent by respondent for all {len(a):,} IDs")


def calculate(df: pd.DataFrame, meta: object, country_names: dict[int, str],
              source: str) -> pd.DataFrame:
    """Compare weighted and unweighted percentages using identical raw respondents."""
    rows = []
    for ccode, group in df.groupby("country", sort=True):
        country = country_names[int(ccode)]
        w = numeric(group.weight)
        n = len(group)
        denom_w = float(w.sum())
        for q in QUESTIONS:
            v = numeric(group[q])
            codes_missing = nonresponse_codes(meta, q)
            masks = {1: v.eq(1), 2: v.eq(2), 3: v.eq(3), 8: v.isin(codes_missing)}
            # Exhaustive and exclusive, so every respondent contributes to one cell.
            if sum(int(mask.sum()) for mask in masks.values()) != n:
                raise AssertionError(f"{source}: question-category partition failure: {country}/{q}")
            for code, mask in masks.items():
                count = int(mask.sum())
                uw = 100.0 * count / n
                ww = 100.0 * float(w[mask].sum()) / denom_w
                rows.append({
                    "source": source, "country": country, "question": q,
                    "code": code, "category": "DK/Refused" if code == 8 else str(code),
                    "sample_n": n, "count_n": count, "sum_weight": denom_w,
                    "unweighted_pct": uw, "weighted_pct": ww,
                    "unweighted_rounded": int(np.floor(uw + .5)),
                    "weighted_rounded": int(np.floor(ww + .5)),
                    "delta_unweighted_minus_weighted_pp": uw - ww,
                })
    result = pd.DataFrame(rows)
    expected_cells = len(country_names) * len(QUESTIONS) * len(CATEGORY_CODES)
    if len(result) != expected_cells:
        raise AssertionError(f"{source}: expected {expected_cells} cells, got {len(result)}")
    for col in ("unweighted_pct", "weighted_pct"):
        totals = result.groupby(["country", "question"])[col].sum()
        if not np.allclose(totals, 100.0, atol=1e-9, rtol=0):
            raise AssertionError(f"{source}: {col} is not an exhaustive 100%-denominator")
    return result


def compare_sources(a: pd.DataFrame, b: pd.DataFrame) -> None:
    key = ["country", "question", "code"]
    comp = a.merge(b, on=key, validate="one_to_one", suffixes=("_sav", "_csv"), indicator=True)
    if len(comp) != len(a) or not comp._merge.eq("both").all():
        raise AssertionError("SAV/CSV calculated cells differ")
    for col in ("unweighted_pct", "weighted_pct", "delta_unweighted_minus_weighted_pp"):
        if not np.allclose(comp[f"{col}_sav"], comp[f"{col}_csv"], rtol=0, atol=1e-9):
            raise AssertionError(f"SAV/CSV disagree in calculated {col}")
    if not comp.count_n_sav.equals(comp.count_n_csv):
        raise AssertionError("SAV/CSV sample counts disagree")
    print("PASS: SAV and CSV weighted/unweighted percentages match for all 192 cells")


def check_uk(result: pd.DataFrame) -> pd.Series:
    uk = result.loc[(result.country == "UK") & (result.question == "ai_heard") & (result.code == 1)]
    if len(uk) != 1:
        raise AssertionError("UK Q30 'A lot' reference missing/duplicated")
    s = uk.iloc[0]
    if int(s.sample_n) != 1006 or int(s.count_n) != 442:
        raise AssertionError("UK Q30 observed 442 of 1006 not reproduced")
    if not np.isclose(s.unweighted_pct, 100 * 442 / 1006, rtol=0, atol=1e-10):
        raise AssertionError("UK Q30 unweighted percentage incorrect")
    if not np.isclose(s.weighted_pct, 40.6852204528972, atol=1e-7, rtol=0):
        raise AssertionError("UK Q30 weighted percentage differs from validated Task 19")
    if int(s.unweighted_rounded) != 44 or int(s.weighted_rounded) != 41:
        raise AssertionError("UK Q30 should round to 44% unweighted vs 41% weighted")
    return s


def test() -> None:
    meta = SimpleNamespace(variable_value_labels={
        "ai_heard": {1: "A lot", 2: "A little", 3: "Nothing at all", 8: "Don't know", 9: "Refused"},
        "ai_cncexc": {1: "Excited", 2: "Concerned", 3: "Equally", 8: "Don't know", 9: "Refused"},
    })
    d = pd.DataFrame({"id": [1, 2, 3, 4, 5], "country": [12] * 5,
                      "weight": [1., 2., 1., 3., 3.],
                      "ai_heard": [1, 1, 2, 8, 9],
                      "ai_cncexc": [3, 2, 1, 8, 9]})
    unchanged = d.copy(deep=True)
    check_input(d, meta, "test", n=5, n_countries=1)
    out = calculate(d, meta, {12: "UK"}, "TEST")
    found = out.loc[(out.question == "ai_heard") & (out.code == 1)].iloc[0]
    assert found.count_n == 2 and found.sample_n == 5
    assert np.isclose(found.unweighted_pct, 40.0)
    assert np.isclose(found.weighted_pct, 30.0)
    assert np.isclose(found.delta_unweighted_minus_weighted_pp, 10.0)
    dk = out.loc[(out.question == "ai_heard") & (out.code == 8)].iloc[0]
    assert dk.count_n == 2 and np.isclose(dk.unweighted_pct, 40.0)
    assert np.isclose(dk.weighted_pct, 60.0)
    assert int(out[out.question.eq("ai_heard")].count_n.sum()) == 5
    pd.testing.assert_frame_equal(d, unchanged)
    print("PASS: both percentages use the full sample, including DK/refused")
    print("PASS: unweighted and weighted numerators/denominators independently verified")
    bad = d.copy(deep=True)
    bad.loc[0, "ai_heard"] = 7
    try:
        check_input(bad, meta, "test", n=5, n_countries=1)
        raise AssertionError("Self-test failed to catch unlabelled response code")
    except AssertionError as e:
        if "unexpected or missing" not in str(e):
            raise
    print("PASS: unexpected response code detected")
    bad_meta = SimpleNamespace(variable_value_labels={**meta.variable_value_labels,
                        "ai_heard": {1:"A lot", 2:"A little", 3:"Nothing", 8:"Not sure", 9:"Refused"}})
    try:
        nonresponse_codes(bad_meta, "ai_heard")
        raise AssertionError("Self-test failed to catch incorrect nonresponse labels")
    except AssertionError as e:
        if "Unexpected Q30/Q31" not in str(e):
            raise
    print("PASS: nonresponse-label mismatch detected")
    changed = d.copy(deep=True)
    changed.loc[0, "weight"] = 9.
    try:
        compare_inputs(d, changed)
        raise AssertionError("Self-test failed to detect a modified survey weight")
    except AssertionError as e:
        if "SAV/CSV respondent-level mismatch" not in str(e):
            raise
    print("PASS: respondent-level weight discrepancy detected")
    print("TASK 20 SELF-TEST PASSED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        test()
        return
    print("=== TASK 20: UNWEIGHTED VS WEIGHTED PEW Q30/Q31 ===")
    for p in [SAV, CSV, REF]:
        if not p.is_file():
            raise FileNotFoundError(f"Required input missing: {p}")
    try:
        import pyreadstat
    except ImportError as exc:
        raise RuntimeError("SAV reader missing. Install with python -m pip install pyreadstat") from exc
    original_sav, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    original_sav.columns = original_sav.columns.str.strip().str.lower()
    original_csv = pd.read_csv(CSV, usecols=["id", "country", "weight", *QUESTIONS], low_memory=False,
                               encoding="utf-8-sig")
    original_csv.columns = original_csv.columns.str.strip().str.lower()
    for q in QUESTIONS:
        print(f"{q}: SAV-labelled DK/refused codes {sorted(nonresponse_codes(meta, q))}")
    check_input(original_sav, meta, "SAV")
    check_input(original_csv, meta, "CSV")
    compare_inputs(original_sav, original_csv)
    country_names = {int(code): name for code, name in labels(meta, "country").items()}
    ref = reference()
    if set(country_names.values()) != set(ref.country):
        raise AssertionError("24 country labels do not match Pew's topline reference")
    a = calculate(original_sav, meta, country_names, "SAV")
    b = calculate(original_csv, meta, country_names, "CSV")
    compare_sources(a, b)
    a = a.merge(ref, on=["country", "question", "code"], how="outer", validate="one_to_one", indicator=True)
    if len(a) != 192 or not a._merge.eq("both").all():
        raise AssertionError("Incomplete comparison against 192 Pew published cells")
    a.drop(columns=["_merge"], inplace=True)
    a.rename(columns={"expected": "pew_published_rounded"}, inplace=True)
    a["published_weighted_match"] = a.weighted_rounded.eq(a.pew_published_rounded)
    if not a.published_weighted_match.all():
        print(a.loc[~a.published_weighted_match, ["country", "question", "code", "weighted_rounded", "pew_published_rounded"]]
              .to_string(index=False))
        raise AssertionError("Weighted percentages no longer match Task 19 reference")
    a["unweighted_rounded_minus_pew_pp"] = a.unweighted_rounded - a.pew_published_rounded
    a["weighted_vs_unweighted_rounding_differs"] = a.unweighted_rounded.ne(a.weighted_rounded)
    uk = check_uk(a)
    print("PASS: weighted calculations reproduce all 192/192 Pew published cells")
    print(f"UK Q30 'A lot': raw n={int(uk.count_n):,} / N={int(uk.sample_n):,}")
    print(f"  UNWEIGHTED: {uk.unweighted_pct:.4f}% -> {int(uk.unweighted_rounded)}%")
    print(f"  WEIGHTED:   {uk.weighted_pct:.4f}% -> {int(uk.weighted_rounded)}% (Pew={int(uk.pew_published_rounded)}%)")
    print(f"  Difference (unweighted - weighted): {uk.delta_unweighted_minus_weighted_pp:+.4f} percentage points")
    print(f"All cells: {int(a.weighted_vs_unweighted_rounding_differs.sum())}/192 round differently without weights")
    by_country = (a.groupby(["country", "question"], as_index=False)
                  .agg(n=("sample_n", "first"),
                       changed_rounded_cells=("weighted_vs_unweighted_rounding_differs", "sum"),
                       mean_abs_shift_pp=("delta_unweighted_minus_weighted_pp", lambda x: float(x.abs().mean())),
                       max_abs_shift_pp=("delta_unweighted_minus_weighted_pp", lambda x: float(x.abs().max()))))
    # No sensitive respondent records in Task 20 outputs.
    OUT.mkdir(parents=True, exist_ok=True)
    a.sort_values(["question", "country", "code"]).to_csv(OUT / "task20_weighting_comparison.csv", index=False)
    by_country.sort_values(["question", "country"]).to_csv(OUT / "task20_country_summary.csv", index=False)
    a.loc[(a.country == "UK") & (a.question == "ai_heard")].sort_values("code").to_csv(
        OUT / "task20_uk_q30_detail.csv", index=False)
    (OUT / "task20_validation_note.txt").write_text(
        "Task 20 — compare unweighted versus weighted Pew Q30/Q31 outcomes.\n"
        "Source: original SAV and CSV, respondent-level matching across all 28,333 IDs.\n"
        "All respondents, including those answering Don't know or Refused, appear in BOTH denominators.\n"
        "SAV-labelled codes 8 and 9 are combined into the DK/Refused summary cell.\n"
        "Unweighted = count(category)/N(country); weighted = sum(weight within category)/sum(weight within country).\n"
        "Published reference: Task 19 official Pew Q30/Q31 topline (192 cells), rounded to nearest integer.\n"
        f"UK Q30 'A lot': n={int(uk.count_n)}/{int(uk.sample_n)}, "
        f"unweighted={uk.unweighted_pct:.6f}% (~44%), weighted={uk.weighted_pct:.6f}% (~41%).\n"
        f"Unweighted minus weighted = {uk.delta_unweighted_minus_weighted_pp:.6f} pp.\n"
        f"All 192 weighted cells agree with Pew; {int(a.weighted_vs_unweighted_rounding_differs.sum())} "
        "cells have different rounded unweighted percentages.\n"
        "Weights are appropriate for main estimates; unweighted percentages are a sensitivity/illustrative comparison.\n"
        "These are descriptive point estimates; do not infer statistical significance from the shift.\n"
        "No original files or pew_clean.csv modified.\n", encoding="utf-8")
    pd.DataFrame([{
        "task": 20,
        "decision": "Use Pew survey weights for main results; show unweighted estimates as a comparison",
        "alternative": "Ignore weights for the main population estimates, or drop DK/refused from the denominator",
        "reason": "Published Pew Q30/Q31 percentages require weights and include DK/refused; the alternative shifts results",
        "effect": f"UK Q30 A lot: 44% unweighted vs 41% weighted, +{uk.delta_unweighted_minus_weighted_pp:.4f} pp exact",
        "rows_removed": 0,
        "status": "PASSED"
    }]).to_csv(OUT / "task20_decision_log_entry.csv", index=False)
    pd.DataFrame([{
        "task": 20,
        "ai_use": "Generated unweighted-versus-weighted comparison and automated validation script",
        "evidence": "Original SAV value labels, original SAV and CSV responses, Task 19 192-cell Pew topline reference",
        "verification": "Respondent-level SAV/CSV match; 192-cell numeric match; UK 442/1006 raw share",
        "result": "PASSED — independently verified weighted and unweighted computations",
        "ai_error": "None identified in Task 20 testing"
    }]).to_csv(OUT / "task20_ai_use_log_entry.csv", index=False)
    print("Saved: task20_weighting_comparison.csv, task20_country_summary.csv,")
    print("       task20_uk_q30_detail.csv, task20_validation_note.txt,")
    print("       task20_decision_log_entry.csv, task20_ai_use_log_entry.csv")
    print("TASK 20 PASSED: UK Q30 44% unweighted vs 41% weighted; 192/192 weighted")
    print("                Pew cells reproduced; SAV/CSV agree; all respondents preserved.")


if __name__ == "__main__":
    main()
