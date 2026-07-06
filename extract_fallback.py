"""
Task 2 – Regex Fallback Extractor

Extracts invoice fields using regular expressions when the LLM
fails to produce valid JSON after retry.
"""

import re
from typing import List

from models import Invoice, LineItem


def first_match(patterns: List[str], text: str) -> str:
    """Return the first regex group that matches any of the patterns."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    return ""


def extract_line_items(text: str) -> List[LineItem]:
    """
    Attempt to extract line items from invoice text.

    Looks for patterns like:
        Description  Qty  UnitPrice  Amount
    or individual item lines with amounts.
    """
    items = []

    # Pattern: description followed by amount on the same line
    # e.g. "Premium Support Plan 8 16,313.00 130,504.00"
    pattern = re.compile(
        r"^(.+?)\s+(\d+)\s+[\d,]+\.\d{2}\s+([\d,]+\.\d{2})$",
        re.MULTILINE,
    )

    for match in pattern.finditer(text):
        description = match.group(1).strip()
        amount_str = match.group(3).replace(",", "")

        # Skip header-like lines
        if description.lower() in ("description", "item", "product", "service"):
            continue

        try:
            items.append(
                LineItem(
                    description=description,
                    amount=float(amount_str),
                )
            )
        except ValueError:
            continue

    # Simpler pattern: "description ... amount"
    if not items:
        simple_pattern = re.compile(
            r"^([A-Za-z][A-Za-z\s&]+?)\s{2,}([\d,]+\.\d{2})$",
            re.MULTILINE,
        )

        for match in simple_pattern.finditer(text):
            description = match.group(1).strip()
            amount_str = match.group(2).replace(",", "")

            skip_words = {
                "total", "grand total", "subtotal", "sub total",
                "tax", "discount", "balance", "amount due",
                "date", "invoice", "billing",
            }

            if description.lower() in skip_words:
                continue

            try:
                items.append(
                    LineItem(
                        description=description,
                        amount=float(amount_str),
                    )
                )
            except ValueError:
                continue

    return items


def extract_with_regex(invoice_text: str) -> Invoice:
    """
    Extract invoice fields using regex as a fallback.

    Returns an Invoice object with best-effort field extraction.
    """

    # --- Invoice Number ---
    invoice_number = first_match(
        [
            r"Billing\s*ID\s*:\s*([A-Za-z0-9\-\/]+)",
            r"Invoice\s*Number\s*:\s*([A-Za-z0-9\-\/]+)",
            r"Invoice\s*No\.?\s*:\s*([A-Za-z0-9\-\/]+)",
            r"Invoice\s*#\s*:\s*([A-Za-z0-9\-\/]+)",
            r"Ref(?:erence)?\s*(?:No\.?|Number)\s*:\s*([A-Za-z0-9\-\/]+)",
        ],
        invoice_text,
    )

    # --- Vendor ---
    vendor = first_match(
        [
            r"Vendor\s*:\s*(.+)",
            r"Supplier\s*:\s*(.+)",
            r"Company\s*:\s*(.+)",
            r"From\s*:\s*(.+)",
            r"Bill\s*From\s*:\s*(.+)",
        ],
        invoice_text,
    )

    # If no vendor label, use the first non-empty line
    if not vendor:
        for line in invoice_text.splitlines():
            line = line.strip()
            if line and not line.lower().startswith(("invoice", "date", "bill")):
                vendor = line
                break

    # --- Invoice Date ---
    invoice_date = first_match(
        [
            r"Date\s*of\s*issue\s*:?\s*([A-Za-z0-9\-\/\., ]+)",
            r"Invoice\s*Date\s*:\s*([A-Za-z0-9\-\/\., ]+)",
            r"Date\s*:\s*([A-Za-z0-9\-\/\., ]+)",
            r"Issued\s*:\s*([A-Za-z0-9\-\/\., ]+)",
        ],
        invoice_text,
    )

    # Clean trailing terms/due-date info from date string
    if invoice_date:
        invoice_date = re.split(
            r",?\s*(?:terms|due|net\s*\d)", invoice_date, flags=re.IGNORECASE
        )[0].strip()

    # --- Total Amount & Currency ---
    total_amount = 0.0
    currency = ""

    amount_patterns = [
        r"Grand\s*Total\s*:\s*([A-Z]{3})\s*([\d,]+\.?\d*)",
        r"Total\s*(?:Amount)?\s*:\s*([A-Z]{3})\s*([\d,]+\.?\d*)",
        r"Amount\s*Due\s*:\s*([A-Z]{3})\s*([\d,]+\.?\d*)",
        r"Grand\s*Total\s*:\s*([\d,]+\.?\d*)\s*([A-Z]{3})",
        r"Total\s*(?:Amount)?\s*:\s*([\d,]+\.?\d*)\s*([A-Z]{3})",
        r"Amount\s*Due\s*:\s*([\d,]+\.?\d*)\s*([A-Z]{3})",
        r"Balance\s*Due\s*:\s*([A-Z]{3})\s*([\d,]+\.?\d*)",
        r"Balance\s*Due\s*:\s*([\d,]+\.?\d*)\s*([A-Z]{3})",
    ]

    for pattern in amount_patterns:
        match = re.search(pattern, invoice_text, re.IGNORECASE)
        if match:
            g1, g2 = match.group(1), match.group(2)

            if g1.isalpha():
                currency = g1.upper()
                total_amount = float(g2.replace(",", ""))
            else:
                total_amount = float(g1.replace(",", ""))
                currency = g2.upper()
            break

    # If no currency found, try standalone currency code
    if not currency:
        curr_match = re.search(r"\b(USD|EUR|GBP|INR|CAD|AUD|JPY)\b", invoice_text)
        if curr_match:
            currency = curr_match.group(1)

    # --- Line Items ---
    line_items = extract_line_items(invoice_text)

    return Invoice(
        invoice_number=invoice_number,
        vendor=vendor,
        invoice_date=invoice_date,
        total_amount=total_amount,
        currency=currency,
        line_items=line_items,
    )