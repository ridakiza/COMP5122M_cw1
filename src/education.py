import pandas as pd
import numpy as np
 
COUNTRY_CODES = {
    "d_educ_canada":      2,
    "d_educ_france":      3,
    "d_educ_germany":     4,
    "d_educ_greece":      5,
    "d_educ_hungary":     6,
    "d_educ_italy":       7,
    "d_educ_netherlands": 8,
    "d_educ_poland":      9,
    "d_educ_spain":       10,
    "d_educ_sweden":      11,
    "d_educ_uk":          12,
    "d_educ_australia":   13,
    "d_educ_india":       14,
    "d_educ_indonesia":   15,
    "d_educ_japan":       16,
    "d_educ_skorea":      17,

    "d_educ_turkey":      19,
}

CODE_TO_COUNTRY = {code: name for name, code in COUNTRY_CODES.items() if code is not None}

#  note it in in range so make sure u increase the end of the range with +1 of ur actual mapping value
EDU_MAP = {
    "d_educ_canada":      {"Low": range(1, 6),  "Medium": range(6, 9),  "High": range(9, 12)},
    "d_educ_france":      {"Low": range(1, 4),  "Medium": range(4, 9),  "High": range(9, 14)},
    "d_educ_germany":     {"Low": range(1, 3),  "Medium": range(3, 5),  "High": range(5, 8)},
    "d_educ_greece":      {"Low": range(1, 5),  "Medium": range(5, 7),  "High": range(7, 13)},
    "d_educ_hungary":     {"Low": range(1, 4),  "Medium": range(4, 9),  "High": range(9, 12)},
    "d_educ_italy":       {"Low": range(1, 4),  "Medium": range(4, 6),  "High": range(6, 9)},
    "d_educ_netherlands": {"Low": range(1, 5),  "Medium": range(5, 8),  "High": range(8, 11)},
    "d_educ_poland":      {"Low": range(1, 4),  "Medium": range(4, 8),  "High": range(8, 11)},
    "d_educ_spain":       {"Low": range(1, 6),  "Medium": range(6, 8),  "High": range(8, 13)},
    "d_educ_sweden":      {"Low": range(1, 4),  "Medium": range(4, 6),  "High": range(6, 11)},
    "d_educ_uk":          {"Low": range(1, 3),  "Medium": range(3, 5),  "High": range(5, 10)},
    "d_educ_australia":   {"Low": range(1, 2),  "Medium": range(2, 3),  "High": range(3, 4)},
    "d_educ_india":       {"Low": range(1, 6),  "Medium": range(6, 7),  "High": range(7, 12)},
    "d_educ_indonesia":   {"Low": range(1, 7),  "Medium": range(7, 8),  "High": range(8, 13)},
    "d_educ_japan":       {"Low": range(1, 4),  "Medium": range(4, 5),  "High": range(5, 10)},
    "d_educ_skorea":      {"Low": range(1, 5),  "Medium": range(5, 7),  "High": range(7, 11)},

    "d_educ_turkey":      {"Low": range(1, 7),  "Medium": range(7,9),  "High": range(9, 12)}
}
 
# Flatten into {country: {code: level}} for fast lookup
LOOKUP = {
    country: {code: level for level, codes in levels.items() for code in codes}
    for country, levels in EDU_MAP.items()
} 

def classify_education(row):
    """Return Low/Medium/High for one respondent, or NaN if missing."""

    key = CODE_TO_COUNTRY.get(row["country"])
    if key is None:
        return np.nan
 
    value = row[key]
    if pd.isna(value):
        return np.nan
 
    return LOOKUP[key].get(int(value), np.nan)

def adjust_germany_vocational(row):
    """For Germany, code 3 depends on vocational training:
       vocational = 1 (Yes) -> Medium
       vocational = 2 (No)  -> Low"""
    
    if CODE_TO_COUNTRY.get(row["country"]) != "d_educ_germany" or row["d_educ_germany"] != 3:
        return row["edu_level"]
 
    vocational = row["d_educ_germany_vocational"]
    if vocational == 1:
        return "Medium"
    else:
        return "Low"

# for this script to work u need the cleaned file in this directory
df = pd.read_csv("pew_clean_p1_provisional.csv") 
 
df["edu_level"] = df.apply(classify_education, axis=1)
df["edu_level"] = df.apply(adjust_germany_vocational, axis=1)

df.to_csv("pew_with_edu.csv", index=False)
 