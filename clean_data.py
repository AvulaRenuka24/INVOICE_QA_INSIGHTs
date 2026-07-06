from pathlib import Path
import pandas as pd
from dateutil import parser

INPUT_FILE = Path("data/extracted_invoices.csv")
OUTPUT_FILE = Path("data/clean_invoices.csv")

USD_RATES = {
    "USD": 1.0,
    "EUR": 1.10,
    "GBP": 1.30,
    "INR": 0.012,
}


def clean_vendor(vendor: str) -> str:
    if pd.isna(vendor):
        return ""
    return " ".join(vendor.upper().split())


def to_iso(date_text: str) -> str:
    try:
        return parser.parse(str(date_text)).date().isoformat()
    except Exception:
        return ""


def convert_to_usd(amount, currency):
    try:
        return round(float(amount) * USD_RATES.get(str(currency).upper(), 1.0), 2)
    except Exception:
        return None


def line_items_match(items, total):
    if not isinstance(items, list):
        return True
    try:
        calc = sum(float(i.get("amount", 0)) for i in items)
        return abs(calc - float(total)) <= 0.01
    except Exception:
        return False


def main():
    if not INPUT_FILE.exists():
        print(f"Missing input file: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    before = len(df)

    if "vendor" in df.columns:
        df["vendor"] = df["vendor"].apply(clean_vendor)

    if "invoice_date" in df.columns:
        df["invoice_date"] = df["invoice_date"].apply(to_iso)

    if "currency" in df.columns and "total_amount" in df.columns:
        df["amount_usd"] = df.apply(
            lambda r: convert_to_usd(r["total_amount"], r["currency"]),
            axis=1
        )

    if "review_status" in df.columns:
        failed = df[df["review_status"].astype(str).str.lower() == "failed"]
        if not failed.empty:
            print("\nFailed Extractions")
            print(failed)

    df = df.drop_duplicates()

    duplicate_subset = [
        c for c in [
            "invoice_number",
            "vendor",
            "invoice_date",
            "total_amount",
            "currency"
        ] if c in df.columns
    ]

    if duplicate_subset:
        df = df.drop_duplicates(subset=duplicate_subset)

    if "line_items" in df.columns and "total_amount" in df.columns:
        mismatches = []
        for _, row in df.iterrows():
            try:
                import ast
                items = ast.literal_eval(str(row["line_items"]))
            except Exception:
                items = []
            if not line_items_match(items, row["total_amount"]):
                mismatches.append(row.get("invoice_number", "UNKNOWN"))

        if mismatches:
            print("\nLine Item Mismatches")
            for inv in mismatches:
                print(inv)

    after = len(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCleaning Summary")
    print(f"Rows Before : {before}")
    print(f"Rows After  : {after}")
    print(f"Removed     : {before-after}")
    print(f"\nSaved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
