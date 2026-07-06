from pathlib import Path
import csv

from pydantic import ValidationError

from llm import generate
from models import Invoice
from extract_fallback import extract_with_regex

PROMPTS_DIR = Path("prompts")
LOG_FILE = Path("logs/llm_calls.log")


def load_prompt(prompt_file: str):
    return (PROMPTS_DIR / prompt_file).read_text(encoding="utf-8")


def clean_json_response(response: str) -> str:
    response = response.strip()

    if response.startswith("```json"):
        response = response[7:]

    if response.startswith("```"):
        response = response[3:]

    if response.endswith("```"):
        response = response[:-3]

    return response.strip()


def call_llm(prompt: str) -> str:

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = generate(
        messages=messages,
        max_tokens=200
    )

    return clean_json_response(response)


def log_result(filename: str, method: str):

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{filename} -> {method}\n")


def extract_invoice(
    invoice_text: str,
    filename: str = "sample",
    prompt_file: str = "extraction_v1_plain.txt"
) -> Invoice:

    prompt = load_prompt(prompt_file).replace("{text}", invoice_text)

    try:

        response = call_llm(prompt)

        invoice = Invoice.model_validate_json(response)

        log_result(filename, "llm")

        return invoice

    except ValidationError as e:

        retry_prompt = (
            prompt
            + "\n\nPrevious response failed validation.\n"
            + str(e)
            + "\nReturn ONLY valid JSON."
        )

        try:

            response = call_llm(retry_prompt)

            invoice = Invoice.model_validate_json(response)

            log_result(filename, "retry")

            return invoice

        except Exception:

            invoice = extract_with_regex(invoice_text)

            log_result(filename, "fallback")

            return invoice


if __name__ == "__main__":

    import pdfplumber

    invoice_folder = Path("data/raw_invoices/invoices_corpus")

    files = [invoice_folder / f"{i}.pdf" for i in range(1, 21)]

    output_file = Path("data/extracted_invoices.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    extracted_rows = []

    for file in files:

        if not file.exists():
            print(f"Skipping missing file: {file.name}")
            continue

        text = ""

        with pdfplumber.open(file) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

        invoice = extract_invoice(
            invoice_text=text,
            filename=file.name,
            prompt_file="extraction_v1_plain.txt"
        )

        print("=" * 60)
        print(file.name)
        print(invoice.model_dump())

        extracted_rows.append({
            "source_file": file.name,
            "invoice_number": invoice.invoice_number,
            "vendor": invoice.vendor,
            "invoice_date": invoice.invoice_date,
            "total_amount": invoice.total_amount,
            "currency": invoice.currency,
            "line_items": str(invoice.line_items)
        })

    with open(output_file, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_file",
                "invoice_number",
                "vendor",
                "invoice_date",
                "total_amount",
                "currency",
                "line_items"
            ]
        )

        writer.writeheader()
        writer.writerows(extracted_rows)

    print("\n" + "=" * 60)
    print(f"Saved {len(extracted_rows)} invoices to:")
    print(output_file)
    print("=" * 60)