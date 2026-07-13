"""
Task 2 – Invoice Extraction

Extracts structured invoice data from PDF text using:
  1. LLM (best prompt v3 from Task 3 evaluation)
  2. Retry with validation error feedback
  3. Regex fallback

Usage:
    python extract.py
"""

import csv
import logging
from pathlib import Path

import pdfplumber
from pydantic import ValidationError

from llm import generate
from models import Invoice
from extract_fallback import extract_with_regex

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path("prompts")
LOG_FILE = Path("logs/llm_calls.log")

# Best prompt selected from Task 3 evaluation
BEST_PROMPT = Path("prompts/extraction_v1_plain.txt")


def load_prompt(prompt_file: str) -> str:
    """Load a prompt template from the prompts directory."""
    return (PROMPTS_DIR / prompt_file).read_text(encoding="utf-8")


def clean_json_response(response: str) -> str:
    """Strip markdown code fences from LLM JSON output."""
    response = response.strip()

    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    return response.strip()


def call_llm(prompt: str, max_tokens: int = 512) -> str:
    """Send a prompt to the LLM and return cleaned JSON."""
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = generate(messages=messages, max_tokens=max_tokens)

    return clean_json_response(response)


def log_result(filename: str, method: str) -> None:
    """Append extraction path to the log file."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{filename} -> {method}\n")


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from all pages of a PDF."""
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)

    return "\n".join(pages)


def extract_invoice(
    invoice_text: str,
    filename: str = "sample",
    prompt_file: str = BEST_PROMPT,
) -> Invoice:
    """
    Extract invoice data using a three-stage pipeline:

    1. LLM extraction with the given prompt
    2. Retry once – includes the validation error in the prompt
    3. Regex fallback

    Parameters
    ----------
    invoice_text : str
        Raw text extracted from the PDF.
    filename : str
        Name of the source file (for logging).
    prompt_file : str
        Prompt template to use (default: best prompt from Task 3).

    Returns
    -------
    Invoice
        Validated Invoice object.
    """

    prompt = load_prompt(prompt_file).replace("{text}", invoice_text)

    # first_error must be declared outside the try/except, because Python
    # deletes an `except ... as name` binding as soon as that block ends.
    first_error = None

    # --- Stage 1: LLM ---
    try:
        response = call_llm(prompt)
        invoice = Invoice.model_validate_json(response)
        log_result(filename, "llm")
        return invoice

    except Exception as e:
        first_error = e
        logger.warning(f"{filename}: LLM attempt 1 failed – {first_error}")

    # --- Stage 2: Retry with error feedback ---
    try:
        retry_prompt = (
            prompt
            + "\n\nPrevious response failed validation.\n"
            + str(first_error)
            + "\nReturn ONLY valid JSON matching the schema exactly."
        )

        response = call_llm(retry_prompt)
        invoice = Invoice.model_validate_json(response)
        log_result(filename, "retry")
        return invoice

    except Exception as retry_error:
        logger.warning(f"{filename}: Retry failed – {retry_error}")

    # --- Stage 3: Regex fallback ---
    invoice = extract_with_regex(invoice_text)
    log_result(filename, "fallback")
    return invoice


# ---------------------------------------------------------------------------
# CLI – Extract from first 20 PDFs and save CSV
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    invoice_folder = Path("data/raw_invoices/invoices_corpus")

    files = sorted(
        invoice_folder.glob("*.pdf"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )[:20]

    output_file = Path("data/extracted_invoices.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    extracted_rows = []

    for file in files:

        if not file.exists():
            print(f"Skipping missing file: {file.name}")
            continue

        text = extract_pdf_text(file)

        if not text.strip():
            print(f"Skipping empty/scanned PDF: {file.name}")
            continue

        invoice = extract_invoice(
            invoice_text=text,
            filename=file.name,
            prompt_file=BEST_PROMPT,
        )

        print("=" * 60)
        print(file.name)
        print(invoice.model_dump())

        extracted_rows.append(
            {
                "source_file": file.name,
                "invoice_number": invoice.invoice_number,
                "vendor": invoice.vendor,
                "invoice_date": invoice.invoice_date,
                "total_amount": invoice.total_amount,
                "currency": invoice.currency,
                "line_items": str(
                    [item.model_dump() for item in invoice.line_items]
                ),
            }
        )

    fieldnames = [
        "source_file",
        "invoice_number",
        "vendor",
        "invoice_date",
        "total_amount",
        "currency",
        "line_items",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted_rows)

    print("\n" + "=" * 60)
    print(f"Saved {len(extracted_rows)} invoices to: {output_file}")
    print("=" * 60)