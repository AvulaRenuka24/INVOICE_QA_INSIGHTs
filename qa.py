import re
from pathlib import Path
from pydantic import BaseModel
from llm import generate
from retriever import search

PROMPT_FILE = Path("prompts/answer_v1.txt")


class QAResponse(BaseModel):
    answer: str
    sources: list[str]
    context_found: bool


def load_prompt(prompt_file=PROMPT_FILE):
    return Path(prompt_file).read_text(encoding="utf-8")


def clean_response(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[1]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def faithfulness_check(answer: str, chunks: list) -> bool:
    """
    Verify that important words from the answer appear
    in at least one retrieved chunk.
    """

    answer_words = [
        word.lower()
        for word in answer.replace(",", " ").split()
        if len(word) > 2
    ]

    for chunk in chunks:
        chunk_text = chunk["chunk"].lower()

        matches = sum(
            1
            for word in answer_words
            if word in chunk_text
        )

        if matches >= max(1, len(answer_words) // 2):
            return True

    return False


# Labels that mark the FINAL amount owed, in priority order — never the
# subtotal. A 0.5B model doesn't reliably follow a written instruction to
# prefer these over "Subtotal", so this is enforced in code, not the prompt.
FINAL_TOTAL_PATTERNS = [
    r"Please\s+remit\s+([A-Z]{0,3}\s?[\d,]+\.\d{2})",
    r"(?:Total\s+Amount\s+Due|Amount\s+Due|Balance\s+Due)\s*:?\s*([A-Z]{0,3}\s?[\d,]+\.\d{2})",
    r"(?<!Sub)Total\s*:?\s*([A-Z]{0,3}\s?[\d,]+\.\d{2})",
]

TOTAL_QUESTION_KEYWORDS = ["total", "amount due", "how much", "owe", "balance", "pay"]


def extract_final_total(context: str):
    """
    Find the amount the invoice itself labels as the final amount to be
    paid (e.g. "Please remit X" / "Amount Due: X"), as opposed to a
    subtotal, tax, or shipping line. Returns None if no such label is
    found in the retrieved context.
    """
    for pattern in FINAL_TOTAL_PATTERNS:
        match = re.search(pattern, context, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def is_total_question(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in TOTAL_QUESTION_KEYWORDS)


def ask(
    question: str,
    prompt_file=PROMPT_FILE
) -> QAResponse:

    chunks = search(question)

    if not chunks:
        return QAResponse(
            answer="I don't know",
            sources=[],
            context_found=False
        )

    context = "\n\n".join(
        chunk["chunk"]
        for chunk in chunks
    )

    sources = sorted(
        set(
            chunk["invoice_number"]
            for chunk in chunks
        )
    )

    prompt = load_prompt(prompt_file)

    prompt = prompt.replace("{question}", question)
    prompt = prompt.replace("{context}", context)
    prompt = prompt.replace("{sources}", ", ".join(sources))

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    answer = clean_response(
        generate(
            messages,
            max_tokens=256
        )
    )

    if answer.strip().lower() == "i don't know":
        return QAResponse(
            answer="I don't know",
            sources=[],
            context_found=False
        )

    if not faithfulness_check(answer, chunks):
        return QAResponse(
            answer="I don't know",
            sources=[],
            context_found=False
        )

    # Deterministic backstop: if this is a "total amount" style question
    # and the context contains an explicitly labeled final amount (e.g.
    # "Please remit GBP 6,061.08") that differs from what the model said,
    # trust the label over the model — a 0.5B model frequently grabs the
    # nearest dollar figure (often "Subtotal") instead of reasoning about
    # which one is the actual amount due.
    if is_total_question(question):
        final_total = extract_final_total(context)
        if final_total and final_total.replace(" ", "") not in answer.replace(" ", ""):
            answer = final_total

    return QAResponse(
        answer=answer,
        sources=sources,
        context_found=True
    )


if __name__ == "__main__":

    while True:

        question = input("\nQuestion (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        result = ask(question)

        print("\nAnswer")
        print(result.answer)

        print("\nSources")
        print(result.sources)

        print("\nContext Found")
        print(result.context_found)