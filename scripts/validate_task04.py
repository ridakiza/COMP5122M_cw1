"""COMP5122M P1 - Task 4: find each variable's missing-value codes from the .sav labels.

Run from the repository root:
    python scripts/validate_task04.py

Why: the same number means different things in different questions
(8 = "Don't know" on Q30, but 8 = "Not sure", a real answer, on Q32).
So missing codes are read from each code's label, never assumed.
Nothing in the data is changed here; Task 10 onwards applies these decisions.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "clean"
SAV = RAW / "Pew Research Center Global Attitudes Spring 2025 Dataset.sav"

# A code counts as "no answer" if its label says Don't know or Refused.
# "Not sure" is deliberately NOT matched: the brief says it is a real answer on Q32.
NONRESPONSE = re.compile(r"don.?t\s*know|refused", re.IGNORECASE)


def canonical_label(label: object) -> str:
    """Lower-case a label, straighten curly apostrophes and squash spaces."""
    return " ".join(str(label).casefold().replace("\u2019", "'").split())


def is_nonresponse(label: object) -> bool:
    """True if a code's label says Don't know or Refused. Used by Tasks 10 onwards.
    'Not sure' is deliberately NOT matched (a real answer on Q32)."""
    return bool(NONRESPONSE.search(canonical_label(label)))

# Expected results for the Part I variables, taken from the data dictionary.
EXPECTED_MISSING = {
    "ai_heard": {8, 9}, "ai_cncexc": {8, 9},
    "aireg_us": {9}, "aireg_china": {9}, "aireg_eu": {9}, "aireg_country": {9},
    "internetfreq": {8, 9}, "age": {98, 99}, "gender_australia": {99},
}
# Real answers that look like missing codes and must be kept.
MUST_KEEP = {
    ("aireg_us", 8): "Not sure", ("aireg_china", 8): "Not sure",
    ("aireg_eu", 8): "Not sure", ("aireg_country", 8): "Not sure",
    ("internetfreq", 6): "does not use the internet",
    ("age", 97): "97 or older", ("gender_australia", 3): "Other",
}


def main() -> None:
    print("=== TASK 4: MISSING-VALUE CODES FROM THE .SAV LABELS ===")
    df, meta = pyreadstat.read_sav(str(SAV), apply_value_formats=False, user_missing=True)
    labels = meta.variable_value_labels

    # 1. Missing codes for every labelled variable, found from the label text.
    rows = []
    for col in df.columns:
        for code, label in labels.get(col, {}).items():
            rows.append({
                "variable": col, "code": int(code), "label": label,
                "is_nonresponse": bool(NONRESPONSE.search(str(label))),
                "n_people": int((df[col] == code).sum()),
            })
    codebook = pd.DataFrame(rows)
    missing = (codebook[codebook.is_nonresponse]
               .groupby("variable").code.apply(lambda s: set(s)).to_dict())

    # 2. SPSS can also declare missing values itself; check whether this file does.
    n_spss = len(meta.missing_ranges) + len(getattr(meta, "missing_user_values", {}) or {})
    print(f"\nSPSS user-missing declarations in the file: {n_spss}")
    print("-> the value labels are the only place the file marks 'no answer'." if n_spss == 0
          else "-> compare these with the label-based codes below.")

    # 3. Part I variables: found codes vs expected, with counts.
    print("\nPart I variables (found from labels | expected | people with each code):")
    for col, expected in EXPECTED_MISSING.items():
        found = missing.get(col, set())
        counts = {c: int((df[col] == c).sum()) for c in sorted(found)}
        print(f"  {col:<17} {sorted(found)} | {sorted(expected)} | {counts}")
        assert found == expected, f"{col}: labels give {sorted(found)}, expected {sorted(expected)}"

    # 4. Real answers that must not be blanked.
    print("\nReal answers kept (look like missing codes, but are not):")
    for (col, code), meaning in MUST_KEEP.items():
        assert code not in missing.get(col, set()), f"{col} code {code} wrongly treated as missing"
        print(f"  {col:<17} {code:<3} = {meaning:<26} ({int((df[col] == code).sum()):,} people)")

    # 5. Labels a person should look at: "Not sure" anywhere, and codes in the data with no label.
    review = codebook[codebook.label.str.contains("not sure", case=False)].copy()
    review["reason"] = "Not sure: decide if it is a real answer"
    stray = []
    for col, lab in labels.items():
        observed = set(df[col].dropna().unique())
        unlabelled = observed - set(lab)
        # Numeric variables (age, household size...) label only their special codes,
        # so most observed values are unlabelled by design: skip those.
        if len(unlabelled) > len(observed) / 2:
            continue
        for code in sorted(unlabelled):
            if len(lab) > 0:
                stray.append({"variable": col, "code": int(code), "label": "",
                              "n_people": int((df[col] == code).sum()),
                              "reason": "Code in data but not in codebook"})
    review = pd.concat([review, pd.DataFrame(stray)], ignore_index=True)

    OUT.mkdir(parents=True, exist_ok=True)
    codebook.to_csv(OUT / "task04_all_value_labels.csv", index=False)
    pd.DataFrame([{"variable": v, "missing_codes": ",".join(map(str, sorted(c)))}
                  for v, c in sorted(missing.items())]).to_csv(OUT / "task04_missing_codes.csv", index=False)
    review.to_csv(OUT / "task04_labels_to_review.csv", index=False)

    print(f"\nVariables with value labels: {codebook.variable.nunique()} of {df.shape[1]}")
    print(f"Variables with at least one missing code: {len(missing)}")
    print(f"Labels to review: {len(review)} ('Not sure' labels + codes missing from the codebook)")
    for r in stray:
        print(f"  stray code: {r['variable']} = {r['code']} ({r['n_people']} people)")
    print(f"\nSaved: task04_all_value_labels.csv, task04_missing_codes.csv, task04_labels_to_review.csv in {OUT}")
    print("\nTASK 4 PASSED: missing codes read from labels; real answers protected. No data changed.")


if __name__ == "__main__":
    main()