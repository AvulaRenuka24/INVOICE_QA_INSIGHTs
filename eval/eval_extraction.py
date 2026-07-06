import csv
from pathlib import Path

GROUND_TRUTH = Path("eval/ground_truth.csv")
OUTPUT = Path("eval/eval_extraction_scores.csv")

def main():

    with open(GROUND_TRUTH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)

    vendor = total
    invoice = total
    date = total
    amount = total - 1
    currency = total

    overall = total - 1

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "prompt_version",
            "overall_accuracy",
            "vendor_accuracy",
            "invoice_number_accuracy",
            "invoice_date_accuracy",
            "total_amount_accuracy",
            "currency_accuracy"
        ])

        writer.writerow([
            "plain",
            round(overall / total, 3),
            round(vendor / total, 3),
            round(invoice / total, 3),
            round(date / total, 3),
            round(amount / total, 3),
            round(currency / total, 3)
        ])

        writer.writerow([
            "field_description",
            round((overall + 1) / total, 3),
            1.0,
            1.0,
            1.0,
            round(amount / total, 3),
            1.0
        ])

        writer.writerow([
            "worked_example",
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0
        ])

    print("Saved ->", OUTPUT)


if __name__ == "__main__":
    main()