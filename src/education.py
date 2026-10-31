import pandas as pd
import numpy as np
 
COUNTRY_CODES = {
    "canada":      2,
    "france":      3,
    "germany":     4,
    "greece":      5,
    "hungary":     6,
    "italy":       7,
    "netherlands": 8,
    "poland":      9,
}

CODE_TO_COUNTRY = {code: name for name, code in COUNTRY_CODES.items() if code is not None}

#  note it in in range so make sure u increase the end of the range with +1 of ur actual mapping value
EDU_MAP = {
    "canada":      {"Low": range(1, 6),  "Medium": range(6, 9),  "High": range(9, 12)},
    "france":      {"Low": range(1, 4),  "Medium": range(4, 9),  "High": range(9, 14)},
    "germany":     {"Low": range(1, 3),  "Medium": range(3, 5),  "High": range(5, 8)},
    "greece":      {"Low": range(1, 5),  "Medium": range(5, 7),  "High": range(7, 13)},
    "hungary":     {"Low": range(1, 4),  "Medium": range(4, 9),  "High": range(9, 12)},
    "italy":       {"Low": range(1, 4),  "Medium": range(4, 6),  "High": range(6, 9)},
    "netherlands": {"Low": range(1, 5),  "Medium": range(5, 8),  "High": range(8, 11)},
    "poland":      {"Low": range(1, 4),  "Medium": range(4, 8),  "High": range(8, 11)},
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
 
    value = row[f"d_educ_{key}"]
    if pd.isna(value):
        return np.nan
 
    return LOOKUP[key].get(int(value), np.nan)

def adjust_germany_vocational(row):
    """For Germany, code 3 depends on vocational training:
       vocational = 1 (Yes) -> Medium
       vocational = 2 (No)  -> Low"""
    
    if CODE_TO_COUNTRY.get(row["country"]) != "germany" or row["d_educ_germany"] != 3:
        return row["edu_level"]
 
    vocational = row["d_educ_germany_vocational"]
    if vocational == 1:
        return "Medium"
    if vocational == 2:
        return "Low"
    return np.nan

# for this script to work u need the cleaned file in this directory
df = pd.read_csv("pew_clean_p1_provisional.csv") 
 
df["edu_level"] = df.apply(classify_education, axis=1)
df["edu_level"] = df.apply(adjust_germany_vocational, axis=1)

df.to_csv("pew_with_edu.csv", index=False)
 