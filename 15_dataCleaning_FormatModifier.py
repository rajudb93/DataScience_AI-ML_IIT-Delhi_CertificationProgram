import pandas as pd
from datetime import datetime

# -----------------------------------
# Your data
# -----------------------------------
data = {
    "name": ["raju dubey", "RAJU dubey", "raju DUBEY", "  aMIT  kumAR ", "jaya", "SANJAY"],
    "date": ["2024-01-05", "05/01/24", "5 Jan 2024", "01-05-2024", "05.01.2024", "2024/01/05"],
    "city": ["chEnnai", "  mUmBAi  ", "DELHI", "koLKata", "  pune", "HYDERABAD"]
}

df = pd.DataFrame(data)

print("=== BEFORE CLEAN ===")
print(df)

# -----------------------------------
# Helper: robust date parser
# -----------------------------------
DATE_FORMATS = [
    "%Y-%m-%d",   # 2024-01-05
    "%d/%m/%y",   # 05/01/24
    "%d/%m/%Y",   # 05/01/2024
    "%d %b %Y",   # 5 Jan 2024
    "%d-%m-%Y",   # 01-05-2024
    "%d.%m.%Y",   # 05.01.2024
    "%Y/%m/%d",   # 2024/01/05
]

def parse_date_value(x):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if not s:
        return pd.NaT

    # Try each known format
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    # Last fallback: let pandas try
    try:
        return pd.to_datetime(s, dayfirst=True, errors="coerce")
    except Exception:
        return pd.NaT

# -----------------------------------
# CONFIG: rules per column
# -----------------------------------
cleaning_rules = {
    "name": {"kind": "string_title"},
    "city": {"kind": "string_title"},
    "date": {
        "kind": "date",
        "output_format": "%d/%m/%Y"   # target: DD/MM/YYYY
    }
}

# -----------------------------------
# Cleaner (in-place)
# -----------------------------------
def clean_dataframe_inplace(df, rules):
    for col, rule in rules.items():
        if col not in df.columns:
            print(f"Skipping '{col}' – not in dataframe")
            continue

        kind = rule.get("kind")
        print(f"Cleaning column '{col}' as '{kind}'")

        # ---- STRING: Title Case ----
        if kind == "string_title":
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
                .str.title()
            )

        # ---- DATE: parse with custom function, then format ----
        elif kind == "date":
            out_fmt = rule.get("output_format", "%d/%m/%Y")

            parsed = df[col].apply(parse_date_value)   # <== custom parser
            df[col] = parsed.dt.strftime(out_fmt)

# -----------------------------------
# Apply cleaner
# -----------------------------------
clean_dataframe_inplace(df, cleaning_rules)

print("\n=== AFTER CLEAN ===")
print(df)
