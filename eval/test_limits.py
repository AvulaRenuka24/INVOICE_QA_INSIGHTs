"""
Task 4 – Model Limits Testing

Tests the extraction pipeline on four challenging edge cases:
  1. Scanned invoice (image-based PDF)
  2. Missing total amount
  3. Foreign currency
  4. Near-duplicate invoices

Compares LLM vs Regex outputs and computes confidence scores.
Disagreement between LLM and Regex = low confidence.

Usage:
    python eval/test_limits.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pdfplumber

from llm import generate, LLMUnavailable
from models import Invoice
from extract import clean_json_response, load_prompt, BEST_PROMPT
from extract_fallback import extract_with_regex

INVOICE_DIR = Path("data/raw_invoices/invoices_corpus")
OUTPUT = Path("eval/failures_table.csv")


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def llm_extract(text: str) -> Invoice | None:
    """Try LLM extraction, return None on failure."""
    if not text.strip():
        return None

    prompt = load_prompt(BEST_PROMPT).replace("{text}", text)
    messages = [{"role": "user", "content": prompt}]

    try:
        response = generate(messages, max_tokens=512)
        response = clean_json_response(response)
        return Invoice.model_validate_json(response)
    except Exception:
        return None


def regex_extract(text: str) -> Invoice | None:
    """Try regex extraction, return None on failure."""
    if not text.strip():
        return None
    try:
        return extract_with_regex(text)
    except Exception:
        return None


def compute_confidence(llm_inv: Invoice | None, regex_inv: Invoice | None) -> str:
    """
    Compute confidence based on agreement between LLM and Regex.

    High   = both agree on key fields
    Medium = partial agreement
    Low    = major disagreement or one/both failed
    """
    if llm_inv is None or regex_inv is None:
        return "Low"

    agreements = 0
    checks = 0

    # Compare vendor
    if llm_inv.vendor and regex_inv.vendor:
        checks += 1
        if llm_inv.vendor.upper() == regex_inv.vendor.upper():
            agreements += 1

    # Compare total_amount
    checks += 1
    if abs(llm_inv.total_amount - regex_inv.total_amount) < 1.0:
        agreements += 1

    # Compare currency
    if llm_inv.currency and regex_inv.currency:
        checks += 1
        if llm_inv.currency.upper() == regex_inv.currency.upper():
            agreements += 1

    # Compare invoice_number
    if llm_inv.invoice_number and regex_inv.invoice_number:
        checks += 1
        if llm_inv.invoice_number == regex_inv.invoice_number:
            agreements += 1

    if checks == 0:
        return "Low"

    ratio = agreements / checks

    if ratio >= 0.75:
        return "High"
    elif ratio >= 0.5:
        return "Medium"
    else:
        return "Low"


def classify_result(
    llm_inv: Invoice | None,
    regex_inv: Invoice | None,
    expected_issue: str,
) -> str:
    """Classify the result as Correct / Wrong / Hallucinated."""
    if llm_inv is None and regex_inv is None:
        return "Wrong"

    inv = llm_inv or regex_inv

    # If the LLM invented data that regex didn't find
    if llm_inv and regex_inv:
        if (
            llm_inv.total_amount > 0
            and regex_inv.total_amount == 0
            and "missing" in expected_issue.lower()
        ):
            return "Hallucinated"

    if inv and inv.total_amount > 0 and inv.vendor:
        return "Correct"

    return "Wrong"


def format_output(inv: Invoice | None) -> str:
    """Format an Invoice for the failures table."""
    if inv is None:
        return "No output"

    parts = []
    if inv.vendor:
        parts.append(f"vendor={inv.vendor}")
    if inv.invoice_number:
        parts.append(f"inv={inv.invoice_number}")
    if inv.total_amount > 0:
        parts.append(f"amount={inv.total_amount}")
    if inv.currency:
        parts.append(f"currency={inv.currency}")

    return "; ".join(parts) if parts else "Empty fields"


# ---------------------------------------------------------------------------
# Edge-Case Test Definitions
# ---------------------------------------------------------------------------

def find_scanned_pdf() -> Path | None:
    """Find a scanned/image-based PDF (large file, no extractable text)."""
    for pdf in sorted(INVOICE_DIR.glob("*.pdf")):
        if pdf.stat().st_size > 50000:  # Image PDFs are typically >50KB
            text = extract_pdf_text(pdf)
            if not text.strip():
                return pdf
    return None


def find_foreign_currency_pdf() -> Path | None:
    """Find a PDF with non-USD currency."""
    for pdf in sorted(INVOICE_DIR.glob("*.pdf")):
        if pdf.stat().st_size < 5000:  # Text-based PDFs are small
            text = extract_pdf_text(pdf)
            if text and any(c in text for c in ["EUR", "GBP", "INR"]):
                return pdf
    return None


def find_near_duplicates() -> tuple[Path, Path] | None:
    """Find two PDFs from the same vendor (near duplicates)."""
    vendors = {}
    for pdf in sorted(INVOICE_DIR.glob("*.pdf"))[:50]:
        if pdf.stat().st_size > 5000:
            continue
        text = extract_pdf_text(pdf)
        if not text:
            continue
        inv = regex_extract(text)
        if inv and inv.vendor:
            key = inv.vendor.upper().strip()
            if key in vendors:
                return (vendors[key], pdf)
            vendors[key] = pdf
    return None


def main():
    print("=" * 60)
    print("Task 4 – Model Limits Testing")
    print("=" * 60)

    results = []

    # --- Test 1: Scanned Invoice ---
    print("\n[Test 1] Scanned Invoice")
    scanned = find_scanned_pdf()
    if scanned:
        text = extract_pdf_text(scanned)
        llm_inv = llm_extract(text)
        regex_inv = regex_extract(text)
        confidence = compute_confidence(llm_inv, regex_inv)
        result = classify_result(llm_inv, regex_inv, "scanned invoice")
        results.append({
            "file": scanned.name,
            "test_case": "Scanned invoice (image PDF)",
            "llm_output": format_output(llm_inv),
            "regex_output": format_output(regex_inv),
            "confidence": confidence,
            "result": result,
        })
        print(f"  File: {scanned.name} | Confidence: {confidence} | Result: {result}")
    else:
        results.append({
            "file": "N/A",
            "test_case": "Scanned invoice (image PDF)",
            "llm_output": "No output (empty text)",
            "regex_output": "No output (empty text)",
            "confidence": "Low",
            "result": "Wrong",
        })
        print("  No scanned PDF found with empty text.")

    # --- Test 2: Missing Total ---
    print("\n[Test 2] Missing Total Amount")
    # Use a known problematic file or simulate
    missing_total_file = INVOICE_DIR / "9.pdf"
    if missing_total_file.exists():
        text = extract_pdf_text(missing_total_file)
        llm_inv = llm_extract(text)
        regex_inv = regex_extract(text)
        confidence = compute_confidence(llm_inv, regex_inv)
        result = classify_result(llm_inv, regex_inv, "missing total")
        results.append({
            "file": missing_total_file.name,
            "test_case": "Missing/corrupt data",
            "llm_output": format_output(llm_inv),
            "regex_output": format_output(regex_inv),
            "confidence": confidence,
            "result": result,
        })
        print(f"  File: {missing_total_file.name} | Confidence: {confidence} | Result: {result}")

    # --- Test 3: Foreign Currency ---
    print("\n[Test 3] Foreign Currency")
    foreign = find_foreign_currency_pdf()
    if foreign:
        text = extract_pdf_text(foreign)
        llm_inv = llm_extract(text)
        regex_inv = regex_extract(text)
        confidence = compute_confidence(llm_inv, regex_inv)
        result = classify_result(llm_inv, regex_inv, "foreign currency")
        results.append({
            "file": foreign.name,
            "test_case": "Foreign currency (non-USD)",
            "llm_output": format_output(llm_inv),
            "regex_output": format_output(regex_inv),
            "confidence": confidence,
            "result": result,
        })
        print(f"  File: {foreign.name} | Confidence: {confidence} | Result: {result}")

    # --- Test 4: Near Duplicates ---
    print("\n[Test 4] Near-Duplicate Invoices")
    dupes = find_near_duplicates()
    if dupes:
        for pdf in dupes:
            text = extract_pdf_text(pdf)
            llm_inv = llm_extract(text)
            regex_inv = regex_extract(text)
            confidence = compute_confidence(llm_inv, regex_inv)
            result = classify_result(llm_inv, regex_inv, "near duplicate")
            results.append({
                "file": pdf.name,
                "test_case": "Near-duplicate invoice",
                "llm_output": format_output(llm_inv),
                "regex_output": format_output(regex_inv),
                "confidence": confidence,
                "result": result,
            })
            print(f"  File: {pdf.name} | Confidence: {confidence} | Result: {result}")

    # --- Save Failures Table ---
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "test_case", "llm_output",
                "regex_output", "confidence", "result",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"Saved failures table → {OUTPUT}")
    print(f"Total test cases: {len(results)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
