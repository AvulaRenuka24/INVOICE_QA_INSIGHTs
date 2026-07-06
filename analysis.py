from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = Path("data/clean_invoices.csv")
CHART_DIR = Path("data/charts")


def main():

    if not INPUT_FILE.exists():
        print(f"Missing {INPUT_FILE}")
        return

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    if "amount_usd" in df.columns:
        amount_col = "amount_usd"
        currency = "USD"
    else:
        amount_col = "total_amount"
        currency = df["currency"].iloc[0]

    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"],
        errors="coerce"
    )

    total = df[amount_col].sum()
    count = len(df)
    average = df[amount_col].mean()
    median = df[amount_col].median()

    vendor = (
        df.groupby("vendor")[amount_col]
        .sum()
        .sort_values(ascending=False)
    )

    monthly = (
        df.groupby(df["invoice_date"].dt.to_period("M"))[amount_col]
        .sum()
    )

    monthly.index = monthly.index.astype(str)

    mom = monthly.diff()

    outliers = df[
        df[amount_col] >
        df[amount_col].mean() + 3 * df[amount_col].std()
    ]

    print("\n===== Statistics =====")
    print(f"Total Spend : {total:.2f} {currency}")
    print(f"Invoice Count : {count}")
    print(f"Average : {average:.2f}")
    print(f"Median : {median:.2f}")

    print("\nTop 5 Vendors")
    print(vendor.head(5))

    print("\nMonth-over-Month Change")
    print(mom)

    print("\nOutliers")
    print(outliers[["invoice_number", amount_col]])

    # ----------------------------
    # Vendor Chart
    # ----------------------------

    plt.figure(figsize=(10, 5))

    top10 = vendor.head(10)

    top10.plot(kind="bar")

    plt.title("Spend per Vendor (Top 10)")
    plt.xlabel("Vendor")
    plt.ylabel(f"Spend ({currency})")
    plt.ylim(bottom=0)

    if len(top10) > 0:
        plt.text(
            0,
            top10.iloc[0],
            "Top Vendor",
            ha="center"
        )

    plt.tight_layout()

    plt.savefig(CHART_DIR / "vendor_spend.png")

    plt.close()

    # ----------------------------
    # Monthly Chart
    # ----------------------------

    plt.figure(figsize=(10, 5))

    x = list(range(len(monthly)))

    plt.plot(x, monthly.values, marker="o")

    plt.xticks(x, monthly.index, rotation=45)

    plt.title("Spend per Month")
    plt.xlabel("Month")
    plt.ylabel(f"Spend ({currency})")

    plt.ylim(bottom=0)

    if len(monthly) > 0:

        peak_index = monthly.values.argmax()

        plt.scatter(
            peak_index,
            monthly.values[peak_index]
        )

        plt.text(
            peak_index,
            monthly.values[peak_index],
            "Peak Month"
        )

    plt.tight_layout()

    plt.savefig(CHART_DIR / "monthly_spend.png")

    plt.close()

    print("\nCharts saved successfully to data/charts/")


if __name__ == "__main__":
    main()
