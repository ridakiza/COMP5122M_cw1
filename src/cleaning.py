"""P1: auditable cleaning of the Spring 2025 Pew Global Attitudes survey.

Default source: original SPSS .sav (pyreadstat), preserving numeric response codes.

Run from the repository root: python -m src.cleaning

Validation-only CSV mode is provided for debugging; it does NOT replace SAV-label
validation. To test CSV mode pass --source csv --labels-json <private-label-map>.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
CLEAN = ROOT / 'data' / 'clean'
SAV = RAW / 'Pew Research Center Global Attitudes Spring 2025 Dataset.sav'
CSV = RAW / 'Pew Research Center Global Attitudes Spring 2025 Dataset CSV.csv'
REFERENCE = CLEAN / 'topline_expected.csv'

N_EXPECTED = 28_333
BANDS = ['18–29', '30–49', '50–64', '65+']
Q_VARS = ['ai_heard', 'ai_cncexc']
COUNTRY_SUFFIX = {'South Korea': 'skorea', 'South Africa': 'safrica'}


def report(name: str, before: int, after: int) -> None:
    print(f'{name:<33} before={before:>6,} | after={after:>6,}')


def load_sav(path: Path = SAV) -> tuple[pd.DataFrame, object]:
    """Read value labels and numeric values without applying value formats."""
    try:
        import pyreadstat
    except ImportError as exc:
        raise RuntimeError('Run: pip install -r requirements.txt') from exc
    if not path.is_file():
        raise FileNotFoundError(f'Missing source file: {path}\nPlace it in data/raw/ with the original name.')
    df, meta = pyreadstat.read_sav(str(path), apply_value_formats=False, user_missing=True)
    df.columns = df.columns.str.lower()  # supplied columns are lowercase already
    print(f'Loaded SAV: {df.shape[0]:,} respondents, {df.shape[1]} variables')
    return df, meta


def load_csv_test(path: Path, labels_json: Path) -> tuple[pd.DataFrame, object]:
    """Diagnostic-only loader. The actual coursework pipeline should load the SAV."""
    import json
    df = pd.read_csv(path, low_memory=False, na_values=[' '], skipinitialspace=True)
    with open(labels_json, encoding='utf-8') as f:
        labels = json.load(f)
    labels = {col: {float(k): label for k, label in entry.items()} for col, entry in labels.items()}
    # Only for local fallback tests; this is not SPSS missing-value metadata.
    meta = SimpleNamespace(variable_value_labels=labels, variable_measure={})
    print('WARNING: CSV diagnostic mode; use SAV metadata for your final submission.')
    return df, meta


def as_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def codes(meta: object, col: str) -> dict[float, str]:
    """Get the response-code labels from SPSS metadata, not from guesses."""
    m = getattr(meta, 'variable_value_labels', {}) or {}
    lookup = m.get(col) or m.get(col.upper()) or {}
    return {float(k): str(v) for k, v in lookup.items()}


def missing_codes(meta: object, col: str) -> set[float]:
    """Labels with don't-know/refused are missing; 'not sure' remains substantive.

    Q32 code 8 is NOT missing: its label is 'Not sure'.
    For other variables this function relies on the actual metadata labels.
    """
    return {k for k, text in codes(meta, col).items()
            if any(term in text.lower().replace('’', "'") for term in
                   ("don't know", 'refused', 'prefer not to say'))}


def selected_columns(df: pd.DataFrame) -> list[str]:
    exact = ['id', 'country', 'weight', 'ai_heard', 'ai_cncexc',
             'age', 'age_australia', 'gender', 'gender_australia', 'internetfreq']
    extras = sorted(c for c in df.columns if c.startswith(('aireg_', 'd_age_', 'd_educ_')))
    keep = [*exact, *extras]
    not_found = [col for col in keep if col not in df.columns]
    if not_found:
        raise KeyError(f'Missing required columns: {not_found}')
    return keep


def country_names(meta: object) -> dict[int, str]:
    mapping = {int(k): v for k, v in codes(meta, 'country').items()}
    if len(mapping) != 24:
        raise ValueError(f'Expected labels for 24 countries in SAV; found {len(mapping)}')
    return mapping


def check_basics(raw: pd.DataFrame, meta: object, log: list[dict]) -> list[tuple]:
    print('\n=== B. BASIC AUDITS ===')
    print('shape', raw.shape, 'countries:', raw.country.nunique())
    assert raw.shape == (N_EXPECTED, 483), f'Unexpected source shape: {raw.shape}'
    assert raw.country.nunique() == 24
    print('country sizes:\n', raw.groupby('country').size().to_string())
    print('duplicate IDs:', raw.id.duplicated().sum(), 'duplicate rows:', raw.duplicated().sum())
    assert not raw.id.duplicated().any()
    assert not raw.duplicated().any()
    assert raw.weight.notna().all(), 'Missing weight values'
    assert raw.weight.gt(0).all(), 'Nonpositive weights present'
    weight_errors = (raw.groupby('country').weight.sum() - raw.groupby('country').size()).abs()
    print('weight min/max:', raw.weight.min(), raw.weight.max())
    print('max |sum weights - country N|:', weight_errors.max())
    assert weight_errors.max() < 1e-5

    # Validate only fully labelled categorical variables (not continuous variables
    # such as exact age, date, weight, ID). The metadata sometimes labels only
    # exceptional values on continuous variables.
    unlabelled = []
    # These have labels only for the exceptional codes (97/98/99 etc.), not
    # for every allowed number. Treat them as numeric range variables.
    partially_labeled_numeric = {'age', 'd_educ_israel_years', 'd_hhpeople', 'd_adults'}
    measure = getattr(meta, 'variable_measure', {}) or {}
    for col in raw.columns:
        if col in partially_labeled_numeric:
            continue
        lab = codes(meta, col)
        kind = str(measure.get(col, '')).lower()
        # In SAV mode use declared measure. In diagnostic CSV mode the
        # dictionary fallback has no measure metadata, so examine all
        # coded fields except explicitly partially labelled numeric ones.
        if (kind and kind not in ('nominal', 'ordinal')) or not lab:
            continue
        observed = set(as_num(raw[col]).dropna().unique().tolist())
        stray = observed - set(lab)
        if stray:
            unlabelled.append((col, sorted(stray)[:10]))
    print('categorical variables with stray/unlabelled codes:', len(unlabelled))
    for item in unlabelled[:12]: print('  ', item)
    if unlabelled:
        log.append(dict(issue='Unlabelled categorical values', people_affected='review',
                        action='Inspect SPSS metadata and identify unexpected values',
                        effect=str(unlabelled[:15])))
    return unlabelled

    country = as_num(raw.country)
    aust = country == 13
    for col in ['age_australia', 'gender_australia']:
        assert as_num(raw.loc[~aust, col]).isna().all(), f'{col} populated outside Australia'
    assert as_num(raw.loc[aust, 'internetfreq']).isna().all(), 'Internet asked in Australia?'
    for col in Q_VARS:
        assert as_num(raw[col]).notna().all(), f'{col} unexpectedly not asked of all'


def compare_sav_csv(raw: pd.DataFrame, csv_path: Path = CSV) -> pd.DataFrame:
    print('\n=== A5. SAV VS CSV ===')
    if not csv_path.is_file():
        print('Skipping comparison: companion CSV not found.')
        return pd.DataFrame()
    check_cols = ['id', 'country', 'weight', 'ai_heard', 'ai_cncexc',
                  'aireg_us', 'age', 'gender', 'internetfreq']
    csv = pd.read_csv(csv_path, usecols=check_cols, low_memory=False,
                      na_values=[' '], skipinitialspace=True)
    results = []
    assert len(csv) == len(raw), 'SAV/CSV row counts differ'
    for col in check_cols:
        l, r = as_num(raw[col]), as_num(csv[col])
        match = np.isclose(l.to_numpy(dtype=float), r.to_numpy(dtype=float),
                           rtol=1e-8, atol=1e-8, equal_nan=True)
        results.append({'variable': col, 'agreement_rows': int(match.sum()),
                        'total_rows': len(raw), 'mismatches': int((~match).sum())})
    result = pd.DataFrame(results)
    print(result.to_string(index=False))
    if result.mismatches.sum():
        print('WARNING: investigate SAV/CSV discrepancies before proceeding')
    return result


def recode_missing(clean: pd.DataFrame, meta: object, col: str) -> int:
    missing = missing_codes(meta, col)
    if not missing:
        raise ValueError(f'No missing-code labels found in SAV for {col}; inspect metadata')
    numeric = as_num(clean[col])
    n = int(numeric.isin(missing).sum())
    before = int(numeric.notna().sum())
    clean[col] = numeric.mask(numeric.isin(missing))
    after = int(clean[col].notna().sum())
    report(col, before, after)
    return n


def band_from_exact_age(age: pd.Series) -> pd.Series:
    return pd.cut(as_num(age), bins=[17, 29, 49, 64, np.inf],
                  labels=BANDS, ordered=True).astype('object')


def compatible_band(label: str) -> str | None:
    """Only recode a follow-up AGE RANGE if the full range fits 1 target band.

    This intentionally rejects e.g. 45-59, 60-74, 50+, 55+ because all
    cross the boundaries 29/49/64. Never use midpoints as exact ages.
    """
    label = str(label).strip().replace('–', '-').replace('—', '-')
    match = re.fullmatch(r'(\d+)\s*(?:-\s*(\d+)|(\+))', label)
    if not match:
        return None
    low = int(match.group(1))
    high = int(match.group(2)) if match.group(2) is not None else float('inf')
    candidates = [(18, 29), (30, 49), (50, 64), (65, float('inf'))]
    for (a, b), band in zip(candidates, BANDS):
        if low >= a and high <= b:
            return band
    return None


def clean_age(clean: pd.DataFrame, raw: pd.DataFrame,
              meta: object, names: dict[int, str], log: list[dict]) -> None:
    print('\n=== C. AGE CLEANING ===')
    original = as_num(raw.age)
    missing = missing_codes(meta, 'age')
    print('Exact-age nonresponses:', int(original.isin(missing).sum()),
          '| capped age 97:', int((original == 97).sum()))
    clean['age'] = original.mask(original.isin(missing))
    clean['age_band'] = band_from_exact_age(clean.age)
    clean['age_band_source'] = np.where(clean.age_band.notna(), 'exact_age', 'unknown')
    clean['age_band_recovered'] = False

    # Australia has age bands, not exact age values.
    aust = as_num(raw.country) == next(k for k, v in names.items() if v == 'Australia')
    austral_mapping = {code: compatible_band(label) for code, label in codes(meta, 'age_australia').items()}
    if not {'18–29', '30–49', '50–64', '65+'}.issubset(set(austral_mapping.values())):
        raise ValueError('Unexpected Australian age-band code labels in SAV')
    aus_bands = as_num(raw.loc[aust, 'age_australia']).map(austral_mapping)
    clean.loc[aust, 'age_band'] = aus_bands
    clean.loc[aust & clean.age_band.notna(), 'age_band_source'] = 'australia_group'

    # Each follow-up column belongs to exactly one country; never pool codes.
    recovered_per_country = {}
    for code, cname in names.items():
        if cname == 'Australia':
            continue
        suffix = COUNTRY_SUFFIX.get(cname, cname.lower().replace(' ', ''))
        col = f'd_age_{suffix}'
        if col not in clean:
            raise KeyError(f'Expected country-specific age follow-up {col}')
        labels = codes(meta, col)
        followup_to_band = {k: compatible_band(lab) for k, lab in labels.items()}
        raw_followup = as_num(raw[col])
        # Strict: country match, missing exact age, valid follow-up mapping.
        mask = ((as_num(raw.country) == code) & clean.age_band.isna())
        proposed = raw_followup.map(followup_to_band)
        can = mask & proposed.notna()
        clean.loc[can, 'age_band'] = proposed[can]
        clean.loc[can, 'age_band_source'] = 'followup_compatible'
        clean.loc[can, 'age_band_recovered'] = True
        if can.any():
            recovered_per_country[cname] = int(can.sum())
        # Follow-up code 8/9 is 'don't know/refused' and is not valid.
        nonresponse = missing_codes(meta, col)
        if nonresponse:
            clean[col] = as_num(clean[col]).mask(as_num(clean[col]).isin(nonresponse))

    recovered = int(clean.age_band_recovered.sum())
    missing_non_aus = int(clean.loc[~aust, 'age_band'].isna().sum())
    missing_aus = int(clean.loc[aust, 'age_band'].isna().sum())
    print('Recovered from country follow-ups:', recovered, recovered_per_country)
    print('Still missing outside Australia:', missing_non_aus,
          '| in Australia:', missing_aus,
          '| total:', int(clean.age_band.isna().sum()))
    assert recovered == 111, f'Expected 111 recoveries; got {recovered}'
    assert missing_non_aus == 183, f'Expected 183 non-Australian missing; got {missing_non_aus}'
    log.append(dict(issue='Nonresponse to exact age', people_affected=int(original.isin(missing).sum()),
                    action='Use only same-country age ranges entirely inside target band',
                    effect=f'{recovered} safely classified; {missing_non_aus} still unclassified outside Australia'))
    if missing_aus:
        log.append(dict(issue='Australian age band unavailable', people_affected=missing_aus,
                        action='Retain missing, do not impute a respondent age',
                        effect=f'Total missing age bands = {int(clean.age_band.isna().sum())}'))

    # Informational conflict: one Israeli respondent has exact age 65 but
    # follow-up category 45-64. Exact reported age takes precedence.
    isr_code = next(k for k, v in names.items() if v == 'Israel')
    mask_isr = ((as_num(raw.country) == isr_code) & (clean.age == 65) &
                (as_num(raw.d_age_israel) == 3))
    conflict = int(mask_isr.sum())
    print('Israel exact-age vs follow-up disagreement:', conflict)
    if conflict:
        log.append(dict(issue='Israel exact age 65 conflicts with 45–64 follow-up',
                        people_affected=conflict, action='Keep exact age 65 => 65+ band',
                        effect='One record retained in 65+ (no fabricated correction)'))


def clean_gender(clean: pd.DataFrame, raw: pd.DataFrame,
                 meta: object, log: list[dict]) -> None:
    """Preserve 'Other' as a third category; use sensitivity check to exclude it."""
    before = int(as_num(raw.gender).notna().sum())
    gender = as_num(raw.gender).combine_first(as_num(raw.gender_australia))
    other = [k for k, v in codes(meta, 'gender_australia').items() if v.lower().strip() == 'other']
    not_answered = missing_codes(meta, 'gender_australia')
    clean['gender_clean'] = gender.mask(gender.isin(not_answered))
    clean['gender_label'] = clean.gender_clean.map({1: 'Male', 2: 'Female', 3: 'Other'})
    after = int(clean.gender_clean.notna().sum())
    report('gender -> gender_clean', before, after)
    if other:
        n_other = int((clean.gender_clean == other[0]).sum())
        print('Australia Other gender:', n_other)
        log.append(dict(issue='Australian respondents selecting Other gender', people_affected=n_other,
                        action='Keep as distinct Other category; test excluding it',
                        effect=f'{n_other} included for gender description, not pooled with male/female'))


def do_clean(raw: pd.DataFrame, meta: object,
             log: list[dict]) -> pd.DataFrame:
    print('\n=== C. CLEANING ===')
    cols = selected_columns(raw)
    clean = raw.loc[:, cols].copy(deep=True)
    names = country_names(meta)
    clean['country_name'] = as_num(clean.country).map(names)
    clean['weight'] = as_num(clean.weight)
    for col in [*Q_VARS, *sorted(c for c in clean if c.startswith('aireg_')), 'internetfreq']:
        n = recode_missing(clean, meta, col)
        log.append(dict(issue=f'{col}: DK/refusal codes', people_affected=n,
                        action='Set labelled missing codes to NaN (Q32 not sure is kept)',
                        effect=f'{int(clean[col].notna().sum())} valid responses retained'))
    # Q32 category labelled 'Not sure' (code 8) must remain present.
    for col in (c for c in clean if c.startswith('aireg_')):
        valid_not_sure = {k for k, v in codes(meta, col).items() if v.lower().strip() == 'not sure'}
        if valid_not_sure:
            assert as_num(clean[col]).isin(valid_not_sure).sum() == as_num(raw[col]).isin(valid_not_sure).sum()
    assert as_num(clean.internetfreq).eq(6).sum() == as_num(raw.internetfreq).eq(6).sum()
    clean_age(clean, raw, meta, names, log)
    clean_gender(clean, raw, meta, log)
    return clean


def check_clean(clean: pd.DataFrame, raw: pd.DataFrame, meta: object) -> None:
    print('\n=== D. POST-CLEAN ASSERTIONS ===')
    assert len(clean) == len(raw) == N_EXPECTED
    assert clean.id.is_unique
    for col in Q_VARS + ['internetfreq', 'age', *[c for c in clean if c.startswith('aireg_')]]:
        invalid = missing_codes(meta, col)
        assert not as_num(clean[col]).isin(invalid).any(), f'Uncleaned missing values in {col}'
    assert clean.loc[clean.age.notna(), 'age_band'].notna().all()
    print('rows:', len(clean), '| missing age band:', clean.age_band.isna().sum(),
          '| safely recovered:', int(clean.age_band_recovered.sum()))
    print('18–29 country sizes (unweighted):')
    tab = clean.loc[clean.age_band == BANDS[0]].groupby('country_name').size().sort_values()
    print(tab.to_string())
    print('internet nonusers retained:', int(as_num(clean.internetfreq).eq(6).sum()))


def weighted_pct(group: pd.DataFrame, col: str, target: int,
                 keep_dk: bool = True, weights: bool = True) -> float:
    """Percentage of responses equal to target among eligible denominators.

    keep_dk=True: include all asked respondents (including DK/refused) in
    denominator, matching Pew's topline. The cleaned numerator can be NaN.
    """
    w = group.weight if weights else pd.Series(1.0, index=group.index)
    valid = pd.Series(True, index=group.index) if keep_dk else group[col].notna()
    den = w.loc[valid].sum()
    return 100 * w.loc[valid & group[col].eq(target)].sum() / den if den else np.nan


def validate_topline(raw: pd.DataFrame, meta: object,
                     ref_path: Path = REFERENCE) -> pd.DataFrame:
    """Check 24 countries × 2 questions × 4 published categories = 192 cells.

    Published category 8 combines DK and refused: calculate it using raw codes
    identified via SPSS labels; do not exclude those people from the denominator.
    """
    print('\n=== D19. PEW WEIGHTED TOPLINE CHECK ===')
    if not ref_path.is_file():
        print(f'No external topline reference at {ref_path}; skipping (NOT VERIFIED).')
        return pd.DataFrame()
    ref = pd.read_csv(ref_path)
    if len(ref) != 192:
        raise ValueError(f'Topline reference should contain 192 values; got {len(ref)}')
    names = country_names(meta)
    rows = []
    for (q, country), grp in ref.groupby(['question', 'country'], sort=False):
        ids = [key for key, value in names.items() if value == country]
        if len(ids) != 1:
            raise ValueError(f'Unrecognized country in reference: {country}')
        df = raw.loc[as_num(raw.country) == ids[0]]
        total = df.weight.sum()
        miss = missing_codes(meta, q)
        for _, item in grp.iterrows():
            cat = int(item.code)
            mask = as_num(df[q]).isin(miss) if cat == 8 else as_num(df[q]).eq(cat)
            percentage = 100 * df.weight.loc[mask].sum() / total
            rounded = int(np.floor(percentage + 0.5))
            rows.append(dict(question=q, country=country, category=cat,
                             computed_pct=rounded, exact_pct=percentage,
                             expected_pct=int(item.expected), match=rounded == int(item.expected)))
    result = pd.DataFrame(rows)
    print(f'Matched: {result.match.sum()}/{len(result)}')
    print(result.loc[~result.match].to_string(index=False) if not result.match.all() else 'All cells match.')
    return result


def report_deff(clean: pd.DataFrame) -> pd.DataFrame:
    """Kish weight-only design effect: n*sum(w²)/(sum(w)²).

    This is NOT the full clustering/stratification survey design effect.
    """
    rows = []
    for country, grp in clean.groupby('country_name'):
        w = grp.weight.to_numpy(dtype=float)
        n = len(w)
        deff = n * float((w ** 2).sum()) / float(w.sum() ** 2)
        rows.append(dict(country=country, n=n, kish_deff=deff, effective_n=n/deff))
    deff = pd.DataFrame(rows).sort_values('kish_deff', ascending=False)
    print('\n=== D22. KISH WEIGHT-ONLY DESIGN EFFECT ===\n', deff.to_string(index=False, float_format=lambda x:f'{x:.2f}'))
    return deff


def check_dk_by_age(raw: pd.DataFrame, clean: pd.DataFrame,
                    meta: object) -> pd.DataFrame:
    """Q31 DK/refused by age: unweighted n/% plus weighted %."""
    bad = missing_codes(meta, 'ai_cncexc')
    out = []
    eligible = clean.loc[clean.age_band.notna()]
    for (country, band), index in eligible.groupby(['country_name', 'age_band']).groups.items():
        g = raw.loc[index]
        w = g.weight
        mask = as_num(g.ai_cncexc).isin(bad)
        weighted_pct = 100 * w.loc[mask].sum() / w.sum()
        unweighted_pct = 100 * mask.mean()
        out.append(dict(country=country, age_band=band, n=len(g),
                        dk_refused_unweighted_pct=unweighted_pct,
                        dk_refused_weighted_pct=weighted_pct))
    result = pd.DataFrame(out)
    print('\n=== D23. Q31 DK/REFUSAL BY AGE (SELECTED) ===')
    print(result[result.country.isin(['Turkey', 'Israel', 'India', 'Kenya', 'South Africa'])]
          .sort_values(['country', 'age_band']).to_string(index=False,
                                                          float_format=lambda v:f'{v:.1f}'))
    return result


def awareness_gap(df: pd.DataFrame, *, weights: bool = True,
                  keep_dk: bool = True, exclude_recovered: bool = False,
                  drop_other_gender: bool = False) -> pd.DataFrame:
    """Example gap: Q30 'a lot' share in 18–29 MINUS 65+ (pp) per country.

    This is a *demonstration* gap for sensitivity checks. If P3/P4 specify
    another metric/outcome, replace this definition in their analysis.
    """
    x = df.copy()
    if exclude_recovered:
        x = x.loc[~x.age_band_recovered]
    if drop_other_gender:
        x = x.loc[x.gender_clean != 3]
    rows = []
    for country, g in x.groupby('country_name'):
        for band in [BANDS[0], BANDS[-1]]:
            chunk = g.loc[g.age_band.eq(band)]
            val = weighted_pct(chunk, 'ai_heard', 1, keep_dk=keep_dk, weights=weights)
            rows.append(dict(country=country, band=band, percent=val))
    wide = pd.DataFrame(rows).pivot(index='country', columns='band', values='percent')
    wide['gap_pp'] = wide[BANDS[0]] - wide[BANDS[-1]]
    return wide[['gap_pp']].reset_index()


def sensitivity_checks(clean: pd.DataFrame) -> pd.DataFrame:
    """E24–27: change one choice at a time and record change in awareness gap."""
    base = awareness_gap(clean).rename(columns={'gap_pp': 'baseline_gap_pp'})
    modes = [
        ('exclude_dk', dict(keep_dk=False)),
        ('unweighted', dict(weights=False)),
        ('without_recovered_age', dict(exclude_recovered=True)),
        ('exclude_australia_other_gender', dict(drop_other_gender=True)),
    ]
    out = []
    for name, kwargs in modes:
        scenario = awareness_gap(clean, **kwargs).rename(columns={'gap_pp': 'scenario_gap_pp'})
        t = base.merge(scenario, on='country')
        t['scenario'] = name
        t['change_pp'] = t.scenario_gap_pp - t.baseline_gap_pp
        out.append(t)
    return pd.concat(out, ignore_index=True)


def screen_all_columns(raw: pd.DataFrame, meta: object) -> pd.DataFrame:
    """G32: rough exploratory association with AI awareness, not a causal model.

    Cramer's V on nonmissing categorical observations; ranked by effect size.
    Requires scipy, and should be treated as an exploratory screen only.
    """
    from scipy.stats import chi2_contingency
    awareness = as_num(raw.ai_heard).mask(as_num(raw.ai_heard).isin(missing_codes(meta, 'ai_heard')))
    rows = []
    for col in raw.columns:
        if col in ('id', 'weight', 'ai_heard', 'survey'):
            continue
        x = as_num(raw[col]); x = x.mask(x.isin(missing_codes(meta, col)))
        if x.dropna().nunique() < 2 or x.dropna().nunique() > 25:
            continue  # skip continuous/high-cardinality items
        joint = pd.DataFrame({'x': x, 'y': awareness}).dropna()
        tab = pd.crosstab(joint.x, joint.y)
        if min(tab.shape) < 2 or len(joint) < 100:
            continue
        chi2 = chi2_contingency(tab, correction=False)[0]
        v = np.sqrt((chi2 / len(joint)) / min(tab.shape[0]-1, tab.shape[1]-1))
        rows.append({'variable': col, 'n': len(joint), 'levels': tab.shape[0],
                     'cramers_v': v})
    return pd.DataFrame(rows).sort_values('cramers_v', ascending=False)


def save_logs(clean: pd.DataFrame, log: list[dict], topline: pd.DataFrame,
              deff: pd.DataFrame, dk: pd.DataFrame, sensitivity: pd.DataFrame,
              comparisons: pd.DataFrame, ai_log: list[dict]) -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN / 'pew_clean.csv', index=False)
    pd.DataFrame(log).to_csv(CLEAN / 'audit_table.csv', index=False)
    if not topline.empty: topline.to_csv(CLEAN / 'topline_check.csv', index=False)
    deff.to_csv(CLEAN / 'kish_deff.csv', index=False)
    dk.to_csv(CLEAN / 'dk_by_age.csv', index=False)
    sensitivity.to_csv(CLEAN / 'sensitivity.csv', index=False)
    if not comparisons.empty: comparisons.to_csv(CLEAN / 'sav_csv_comparison.csv', index=False)
    pd.DataFrame(ai_log, columns=['task', 'expected', 'observed', 'investigation']).to_csv(
        CLEAN / 'ai_log.csv', index=False)
    decisions = [
        ('Source file', 'SAV with numeric codes & metadata', 'CSV only', 'SAV retains original SPSS metadata'),
        ('AI Q30/Q31 DK/refusal', 'Missing for respondent-level models', 'Keep as response',
         'Use raw codes when matching published weighted toplines'),
        ('AI trust Q32 Not sure', 'Retain category 8', 'Set to missing',
         'Not sure is a substantive response, not DK'),
        ('Internet frequency 6', 'Keep nonusers', 'Treat as missing', 'Nonusers are valid respondents'),
        ('Age 97+', 'Keep at 97 for age-band assignment', 'Discard 97', 'Categorized 65+'),
        ('Age follow-up', 'Recover only unambiguous bands', 'Impute age midpoint',
         'Avoids choosing arbitrary ages from incompatible country-specific bands'),
        ('Israeli age disagreement', 'Keep exact age 65', 'Override using follow-up',
         'Exact reported age is more specific than follow-up band'),
        ('Australian gender Other', 'Keep separate category', 'Set missing',
         'Sensitivity check shows effect of excluding 21 respondents'),
        ('Weights', 'Country-normalized Pew weights', 'Unweighted',
         'Official population toplines reproduced using weights'),
        ('Topline DK denominator', 'All respondents including DK', 'Valid-only denominator',
         'Official published Q30/Q31 percentages use all respondents'),
        ('Education', 'Pass country education fields through unchanged', 'Premature pooled mapping',
         'Wait for P2 education recode on 3 Nov'),
    ]
    pd.DataFrame(decisions, columns=['decision', 'chosen', 'alternative', 'rationale_or_effect']).to_csv(
        CLEAN / 'decision_log.csv', index=False)
    print('\nWrote cleaned dataset + local reports to:', CLEAN)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--source', choices=['sav', 'csv'], default='sav')
        parser.add_argument('--csv-path', type=Path, default=CSV)
        parser.add_argument('--labels-json', type=Path)
        parser.add_argument('--scan-all', action='store_true', help='Run exploratory 483-column association scan')
        args = parser.parse_args()
    if args.source == 'sav':
        raw, meta = load_sav()
    else:
        if args.labels_json is None:
            raise ValueError('CSV fallback requires --labels-json derived from your local dictionary')
        raw, meta = load_csv_test(args.csv_path, args.labels_json)
    assert list(raw.columns) == list(dict.fromkeys(raw.columns)), 'Duplicate column names'
    log, ai_log = [], []
    stray_codes = check_basics(raw, meta, log)
    if stray_codes:
        ai_log.append(dict(task=7, expected='No unlabelled codes',
                           observed=str(stray_codes[:10]),
                           investigation='Check SAV labels and source questionnaire. France d_income2_france value 98 may be a codebook inconsistency; outside P1 Part I.'))
    comparisons = compare_sav_csv(raw) if args.source == 'sav' else pd.DataFrame()
    clean = do_clean(raw, meta, log)
    check_clean(clean, raw, meta)
    top = validate_topline(raw, meta)
    if not top.empty and not top.match.all():
        ai_log.append(dict(task=19, expected='192/192', observed=f'{top.match.sum()}/192',
                           investigation='Check weights, coding, denominator, source file, rounding'))
    india_young = clean.loc[(clean.country_name == 'India') & (clean.age_band == BANDS[0])]
    if len(india_young) != 1083:
        ai_log.append(dict(task=21, expected=1083, observed=len(india_young),
                           investigation='India 18–29: 1083 with exact ages + 2 recovered from age follow-up = 1085'))
    if int(clean.age_band.isna().sum()) != 183:
        ai_log.append(dict(task=18, expected=183, observed=int(clean.age_band.isna().sum()),
                           investigation='183 non-Australian missing + 2 Australian missing = 185 overall'))
    deff = report_deff(clean)
    dk = check_dk_by_age(raw, clean, meta)
    sensitivity = sensitivity_checks(clean)
    print('\n=== E24–27. SENSITIVITY SUMMARY (mean absolute gap change in pp) ===')
    print(sensitivity.groupby('scenario').change_pp.apply(lambda x:x.abs().mean()).to_string())
    print('\n=== G33. PHONE SAMPLE LABELS ===')
    print(codes(meta, 'phone_sample'))
    if args.scan_all:
        try:
            screen_all_columns(raw, meta).to_csv(CLEAN / 'associations_screen.csv', index=False)
            print('Saved categorical association scan.')
        except ImportError:
            print('Install scipy to run full association scan.')
    save_logs(clean, log, top, deff, dk, sensitivity, comparisons, ai_log)
    print('Next: open notebooks/01_data_quality.ipynb; share CSV via Teams, NOT GitHub.')


if __name__ == '__main__':
    main()