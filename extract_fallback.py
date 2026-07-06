import re

from models import Invoice


def first_match(patterns, text):
    """
    Return the first regex group that matches.
    """
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    return ""


def extract_with_regex(invoice_text: str) -> Invoice:
    """
    Extract invoice fields using regex as a fallback.
    """

    invoice_number = first_match([
        r"Billing\s*ID\s*:\s*(.+)",
        r"Invoice\s*Number\s*:\s*(.+)",
        r"Invoice\s*No\.?\s*:\s*(.+)",
        r"Invoice\s*#\s*:\s*(.+)",
    ], invoice_text)

    vendor = first_match([
        r"Vendor\s*:\s*(.+)",
        r"Supplier\s*:\s*(.+)",
    ], invoice_text)

    # If no vendor label exists, use the first non-empty line
    if not vendor:
        for line in invoice_text.splitlines():
            line = line.strip()
            if line:
                vendor = line
                break

    invoice_date = first_match([
        r"Date\s*of\s*issue\s*:?\s*(.+)",
        r"Invoice\s*Date\s*:\s*(.+)",
        r"Date\s*:\s*(.+)",
    ], invoice_text)

    total_amount = 0.0
    currency = ""

    patterns = [
        r"Grand\s*Total\s*:\s*([A-Z]{3})\s*([\d,]+\.\d{2})",
        r"Total\s*:\s*([A-Z]{3})\s*([\d,]+\.\d{2})",
        r"Grand\s*Total\s*:\s*([\d,]+\.\d{2})\s*([A-Z]{3})",
        r"Total\s*:\s*([\d,]+\.\d{2})\s*([A-Z]{3})",
    ]

    for pattern in patterns:
        match = re.search(pattern, invoice_text, re.IGNORECASE)

        if match:
            g1 = match.group(1)
            g2 = match.group(2)

            if g1.isalpha():
                currency = g1
                total_amount = float(g2.replace(",", ""))
            else:
                total_amount = float(g1.replace(",", ""))
                currency = g2

            break

    return Invoice(
        invoice_number=invoice_number,
        vendor=vendor,
        invoice_date=invoice_date,
        total_amount=total_amount,
        currency=currency,
        line_items=[]
    )