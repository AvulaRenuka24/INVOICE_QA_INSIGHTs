from pathlib import Path
import hashlib
import re

import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/raw_invoices/invoices_corpus")
DB_DIR = Path("data/chroma_db")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_or_create_collection(
    name="invoice_chunks",
    metadata={"hnsw:space": "cosine"}
)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def extract_pdf_text(pdf_path: Path) -> str:
    text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text.append(page_text)

    return "\n".join(text)


def invoice_number_from_text(text: str, default_name: str) -> str:

    patterns = [
        # Labelled formats
        r"Billing\s*ID\s*[:#]?\s*([A-Za-z0-9/\-]+)",
        r"Invoice\s*Number\s*[:#]?\s*([A-Za-z0-9/\-]+)",
        r"Invoice\s*No\.?\s*[:#]?\s*([A-Za-z0-9/\-]+)",
        r"Invoice\s*#\s*([A-Za-z0-9/\-]+)",
        r"Invoice\s*ID\s*[:#]?\s*([A-Za-z0-9/\-]+)",
        r"Reference\s*[:#]?\s*([A-Za-z0-9/\-]+)",

        # Common invoice number formats
        r"\b\d{4}/INV/\d+\b",      # 2026/INV/83490
        r"\b\d+/INV\b",            # 27727/INV
        r"\bINV-\d+\b",            # INV-93354
        r"\b\d+-INV\b",            # 66829-INV
        r"\bINV\d+\b",             # INV12345
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            if match.lastindex:
                return match.group(1).strip()
            else:
                return match.group(0).strip()

    # No pattern matched — don't silently return the bare filename stem,
    # since that can collide with a real invoice number (e.g. a genuine
    # invoice numbered "84"). Prefix it so it's unmistakably a fallback
    # and easy to find/flag for manual review later.
    return f"UNRESOLVED-{default_name}"


def upsert_invoice(pdf_path: Path):

    text = extract_pdf_text(pdf_path)

    if not text.strip():
        print(f"Skipping empty PDF: {pdf_path.name}")
        return

    chunks = chunk_text(text)

    if len(chunks) == 0:
        print(f"No chunks generated for {pdf_path.name}")
        return

    invoice_number = invoice_number_from_text(
        text,
        pdf_path.stem
    )
    if invoice_number.startswith("UNRESOLVED-"):
        print("\nFAILED:", pdf_path.name)
        print(text[:1000])

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for index, chunk in enumerate(chunks):

        embedding = model.encode(chunk)

        ids.append(f"{invoice_number}#{index}")

        documents.append(chunk)

        embeddings.append(embedding.tolist())

        metadatas.append(
            {
                "invoice_number": invoice_number,
                "source_file": pdf_path.name,
                "chunk_index": index,
                "sha256": hashlib.sha256(
                    chunk.encode("utf-8")
                ).hexdigest()
            }
        )

    if len(embeddings) == 0:
        print(f"No embeddings for {pdf_path.name}")
        return

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


def build_index():

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDFs")

    for pdf in pdf_files:

        print(f"Indexing {pdf.name}")

        try:
            upsert_invoice(pdf)

        except Exception as e:
            print(f"Skipped {pdf.name}: {e}")

    print("\nIndex completed successfully.")
    print(f"Total indexed chunks: {collection.count()}")

def search(question: str, top_k: int = 4):

    print("\n==============================")
    print("Question:", question)
    print("==============================")

    print("Collection count:", collection.count())

    query_embedding = model.encode(question).tolist()

    # If the question names a specific invoice number, filter to that
    # invoice's chunks BEFORE ranking by similarity, instead of ranking
    # across the whole corpus and hoping the right invoice wins. Without
    # this, a generic question like "what is the total amount?" returns
    # the top-K most similar chunks from potentially K different invoices,
    # because nothing in the question anchors it to one document.
    named_invoice = invoice_number_from_text(question, default_name="")
    # invoice_number_from_text() always returns *something* now (it falls
    # back to "UNRESOLVED-<default_name>" instead of a bare empty string),
    # so a real match must both be non-empty AND not be that fallback marker.
    named_invoice = "" if named_invoice.startswith("UNRESOLVED-") else named_invoice

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }
    if named_invoice:
        query_kwargs["where"] = {"invoice_number": named_invoice}
        print(f"Question names invoice '{named_invoice}' -> filtering search to that invoice only")

    results = collection.query(**query_kwargs)

    print("\nRAW CHROMADB RESULTS")
    print(results)

    output = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if len(documents) == 0:
        print("\n❌ No documents retrieved from ChromaDB")
        return []

    print(f"\nRetrieved {len(documents)} chunks\n")

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        score = round(1 / (1 + distance), 4)

        print("--------------------------------")
        print("Invoice :", metadata["invoice_number"])
        print("Score   :", score)
        print("Preview :")
        print(document[:200])
        print("--------------------------------")

        output.append(
            {
                "invoice_number": metadata["invoice_number"],
                "source_file": metadata["source_file"],
                "score": score,
                "chunk": document,
            }
        )

    return output



if __name__ == "__main__":

    build_index()

    print("\nSearch Test\n")

    results = search("What is the total amount?")

    for result in results:
        print(result)

    print("\nFirst 10 metadata records:\n")

    sample = collection.get(limit=10)

    for meta in sample["metadatas"]:
        print(meta)

results = collection.get(
    where={"invoice_number": "2026/INV/83490"}
)

print(results)