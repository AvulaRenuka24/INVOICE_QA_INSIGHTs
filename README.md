# Invoice Q&A and Insights: Prompt Engineering & Data Analysis

## Project Overview

Invoice Q&A and Insights is an AI-powered invoice processing system that extracts structured information from PDF invoices using a local Large Language Model (LLM). The project also supports semantic search, question answering, data cleaning, and business analytics.

The system combines Prompt Engineering, Retrieval-Augmented Generation (RAG), ChromaDB, and Data Analysis to automate invoice processing and generate useful business insights.

---

## Objectives

- Extract invoice details from PDF invoices using a local LLM.
- Validate extracted data using Pydantic.
- Retry extraction on validation failure.
- Use Regex as a fallback extractor.
- Compare multiple prompt versions.
- Build a semantic retriever using ChromaDB.
- Answer invoice-related questions using retrieved context.
- Clean and normalize invoice data.
- Generate statistical summaries and visualizations.

---

## Tech Stack

- Python
- Transformers
- Qwen/Qwen2.5-0.5B-Instruct
- pdfplumber
- Pydantic
- Regular Expressions (Regex)
- Sentence Transformers
- all-MiniLM-L6-v2
- ChromaDB
- Pandas
- Matplotlib

---

## Project Structure

```
INVOICE_QA_INSIGHTs/

├── llm.py
├── extract.py
├── extract_fallback.py
├── models.py
├── retriever.py
├── qa.py
├── clean_data.py
├── analysis.py
│
├── prompts/
│     extraction_v1_plain.txt
│     extraction_v2_field_desc.txt
│     extraction_v3_worked_example.txt
│     answer_v1.txt
│     answer_v2.txt
│
├── eval/
│     ground_truth.csv
│     qa_questions.json
│     eval_extraction.py
│     eval_answers.py
│     test_limits.py
│
└── data/
      Charts/
      chroma_db
      raw_invoices
      clean_invoices.csv
      extracted_invoices.csv


## Workflow

PDF Invoices (525)
        │
        ▼
PDF Text Extraction (pdfplumber)
        │
        ▼
Qwen2.5 LLM
        │
        ├──────────────┐
        │              │
        ▼              ▼
JSON Validation     Retry
        │              │
        └──────┬───────┘
               ▼
Regex Fallback
               │
               ▼
Structured Invoice Data
               │
               ▼
Dataset Cleaning
               │
               ▼
ChromaDB + MiniLM Embeddings
               │
               ▼
Semantic Search
               │
               ▼
Question Answering
               │
               ▼
Statistics + Charts


## Features

- Local LLM using Qwen2.5-0.5B-Instruct
- Structured invoice extraction with Pydantic validation
- Automatic retry on validation failure
- Regex fallback extractor
- Prompt engineering with three prompt versions
- Automatic prompt evaluation using ground truth
- Semantic retrieval using ChromaDB + all-MiniLM-L6-v2
- Question Answering with source citations
- Faithfulness check to reduce hallucinations
- Dataset cleaning and normalization
- Statistical analysis and visualization




## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the LLM

```bash
python llm.py "Reply with the word PONG"
```

### Extract Invoice Data

```bash
python extract.py
```

### Build Retriever

```bash
python retriever.py
```

### Question Answering

```bash
python qa.py
```

### Clean Dataset

```bash
python clean_data.py
```

### Generate Statistics and Charts

```bash
python analysis.py
```
## Dataset

- 525 invoice PDFs
- 20 manually labelled invoices for evaluation
- ChromaDB vector database

## Outputs

The project generates:

Generated Outputs

- extracted_invoices.csv
- clean_invoices.csv
- eval_extraction_scores.csv
- eval_answers_scores.csv
- failures_table.csv
- vendor_spend.png
- monthly_spend.png



## Future Improvements

- Improve retrieval accuracy using metadata-aware search.
- Support scanned invoices with OCR.
- Add FastAPI REST endpoints.
- Build a web interface using Streamlit or React.
- Deploy using Docker.
- Add unit tests and CI/CD pipeline.

---

## Author

Renuka Avula

AI/ML Internship Project – Invoice Q&A and Insights