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
│
├── analysis.py
├── clean_data.py
├── extract.py
├── extract_fallback.py
├── llm.py
├── models.py
├── qa.py
├── retriever.py
├── requirements.txt
│
├── prompts/
│   ├── extraction_v1_plain.txt
│   ├── extraction_v2_field_description.txt
│   ├── extraction_v3_worked_example.txt
│   ├── answer_v1.txt
│   └── answer_v2.txt
│
├── eval/
│   ├── ground_truth.csv
│   ├── qa_questions.json
│   ├── eval_extraction.py
│   ├── eval_answers.py
│   ├── eval_extraction_scores.csv
│   ├── eval_answers_scores.csv
│   └── failures_table.csv
│
├── data/
│   ├── raw_invoices/
│   ├── extracted_invoices.csv
│   ├── clean_invoices.csv
│   └── charts/
│       ├── vendor_spend.png
│       └── monthly_spend.png
│
└── logs/
    └── llm_calls.log
```

---

## Workflow

```
PDF Invoices
        │
        ▼
Text Extraction (pdfplumber)
        │
        ▼
Local LLM (Qwen2.5)
        │
        ▼
JSON Validation (Pydantic)
        │
        ▼
Retry on Failure
        │
        ▼
Regex Fallback
        │
        ▼
Extracted Invoice Data
        │
        ▼
Prompt Evaluation
        │
        ▼
Retriever (ChromaDB)
        │
        ▼
Question Answering
        │
        ▼
Data Cleaning
        │
        ▼
Statistical Analysis
        │
        ▼
Charts & Insights
```

---

## Features

- Local LLM-based invoice extraction
- Retry mechanism for invalid responses
- Regex fallback extraction
- Prompt engineering and evaluation
- Semantic search using embeddings
- Retrieval-Augmented Question Answering (RAG)
- Data cleaning and normalization
- Duplicate detection
- Currency conversion
- Business analytics
- Visualization using Matplotlib

---

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

---

## Outputs

The project generates:

- Extracted invoice dataset
- Clean invoice dataset
- Vendor spend chart
- Monthly spend chart
- Prompt evaluation reports
- Question answering evaluation reports
- Failure analysis report
- LLM execution logs

---

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