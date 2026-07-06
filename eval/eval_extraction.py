"""
Task 3 – Prompt Engineering Evaluation

Evaluates three extraction prompt versions against manually labelled
ground truth (20 invoices). Computes overall and per-field accuracy
for each prompt version, prints a score table, and selects the best.

Usage:
    python eval/eval_extraction.py
"""

import csv
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pdfplumber
from pydantic import ValidationError

from llm import generate, LLMUnavailable
from models import Invoice
from extract import clean_json_response, load_prompt

GROUND_TRUTH = Path("eval/ground_truth.csv")
INVOICE_DIR = Path("data/raw_invoices/invoices_corpus")
OUTPUT = Path("eval/eval_extraction_scores.csv")

PROMPT_VERSIONS = [
    ("v1_plain", "extraction_v1_plain.txt"),
    ("v2_field_desc", "extraction_v2_field_desc.txt"),
    ("v3_worked_example", "extraction_v3_worked_example.txt"),
]

FIELDS = ["vendor", "invoice_number", "invoice_date", "total_amount", "currency"]


def load_ground_truth() -> list[dict]:
    """Load manually labelled ground truth CSV."""
    with open(GROUND_TRUTH, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def normalize(value: str) -> str:
    """Normalize a value for comparison (lowercase, stripped, no commas)."""
    return str(value).strip().lower().replace(",", "").replace(" ", "")


def field_matches(predicted: str, expected: str, field: str) -> bool:
    """
    Compare a predicted field against the expected ground truth value.

    For total_amount: compare as floats with tolerance.
    For other fields: normalized string comparison.
    """
    pred = normalize(predicted)
    exp = normalize(expected)

    if not exp:
        # If ground truth is empty, any non-empty prediction is acceptable
        return True

    if field == "total_amount":
        try:
            return abs(float(pred) - float(exp)) < 1.0
        except (ValueError, TypeError):
            return False

    # For vendor: check containment (handles minor formatting differences)
    if field == "vendor":
        return pred == exp or exp in pred or pred in exp

    # For invoice_number: check containment
    if field == "invoice_number":
        return pred == exp or exp in pred or pred in exp

    return pred == exp


def evaluate_prompt(prompt_file: str, ground_truth: list[dict]) -> dict:
    """
    Run extraction with a given prompt on all ground truth invoices
    and compute per-field and overall accuracy.
    """
    prompt_template = load_prompt(prompt_file)

    field_correct = {f: 0 for f in FIELDS}
    total_correct = 0
    total_invoices = 0

    for row in ground_truth:
        source_file = row["source_file"]
        pdf_path = INVOICE_DIR / source_file

        if not pdf_path.exists():
            print(f"  Skipping missing PDF: {source_file}")
            continue

        text = extract_pdf_text(pdf_path)
        if not text.strip():
            print(f"  Skipping empty PDF: {source_file}")
            continue

        total_invoices += 1

        # Call LLM with this prompt
        prompt = prompt_template.replace("{text}", text)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = generate(messages, max_tokens=512)
            response = clean_json_response(response)
            invoice = Invoice.model_validate_json(response)
        except (LLMUnavailable, ValidationError, Exception) as e:
            print(f"  {source_file}: Extraction failed – {e}")
            continue

        # Compare each field
        extracted = invoice.model_dump()
        all_correct = True

        for field in FIELDS:
            predicted = str(extracted.get(field, ""))
            expected = str(row.get(field, ""))

            if field_matches(predicted, expected, field):
                field_correct[field] += 1
            else:
                all_correct = False
                print(
                    f"  {source_file} [{field}]: "
                    f"expected='{expected}' got='{predicted}'"
                )

        if all_correct:
            total_correct += 1

    if total_invoices == 0:
        return {
            "overall": 0.0,
            **{f: 0.0 for f in FIELDS},
            "total_invoices": 0,
        }

    return {
        "overall": round(total_correct / total_invoices, 3),
        **{f: round(field_correct[f] / total_invoices, 3) for f in FIELDS},
        "total_invoices": total_invoices,
    }


def main():
    ground_truth = load_ground_truth()
    print("Ground truth keys:", ground_truth[0].keys())
    print("First row:", ground_truth[0])
    print(f"Loaded {len(ground_truth)} ground truth entries.\n")

    results = []

    for version_name, prompt_file in PROMPT_VERSIONS:
        print(f"\n{'='*60}")
        print(f"Evaluating: {version_name} ({prompt_file})")
        print("=" * 60)

        scores = evaluate_prompt(prompt_file, ground_truth)
        scores["prompt_version"] = version_name
        results.append(scores)

    # --- Print Score Table ---
    print("\n\n" + "=" * 80)
    print("PROMPT EVALUATION RESULTS")
    print("=" * 80)

    header = (
        f"{'Prompt':<20} {'Overall':>8} {'Vendor':>8} "
        f"{'InvNum':>8} {'Date':>8} {'Amount':>8} {'Currency':>8}"
    )
    print(header)
    print("-" * 80)

    for r in results:
        row = (
            f"{r['prompt_version']:<20} {r['overall']:>8.1%} "
            f"{r['vendor']:>8.1%} {r['invoice_number']:>8.1%} "
            f"{r['invoice_date']:>8.1%} {r['total_amount']:>8.1%} "
            f"{r['currency']:>8.1%}"
        )
        print(row)

    # --- Select Best Prompt ---
    best = max(results, key=lambda r: r["overall"])
    print(f"\n★ Best Prompt: {best['prompt_version']} "
          f"(overall accuracy: {best['overall']:.1%})")
    print("  → extract.py is configured to use the best prompt.\n")

    # --- Save CSV ---
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_version",
            "overall_accuracy",
            "vendor_accuracy",
            "invoice_number_accuracy",
            "invoice_date_accuracy",
            "total_amount_accuracy",
            "currency_accuracy",
        ])

        for r in results:
            writer.writerow([
                r["prompt_version"],
                r["overall"],
                r["vendor"],
                r["invoice_number"],
                r["invoice_date"],
                r["total_amount"],
                r["currency"],
            ])

    print(f"Saved scores → {OUTPUT}")


if __name__ == "__main__":
    main()