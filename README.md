# Invoice Q&A and Insights: Prompt Engineering & Data Analysis

## Project Overview

Invoice Q&A and Insights is an AI-powered invoice processing system that extracts structured information from PDF invoices using a local Large Language Model (LLM). The project also supports semantic search, question answering, dataset cleaning, prompt evaluation, and business analytics.

The system combines Prompt Engineering, Retrieval-Augmented Generation (RAG), ChromaDB, FastAPI, and Data Analysis to automate invoice processing and generate meaningful business insights.

---

## Objectives

- Extract invoice details from PDF invoices using a local LLM.
- Validate extracted data using Pydantic.
- Retry extraction on validation failure.
- Use Regex as a fallback extractor.
- Compare multiple extraction prompts.
- Build a semantic retriever using ChromaDB.
- Answer invoice-related questions using retrieved context.
- Clean and normalize invoice data.
- Generate statistical summaries and visualizations.
- Provide REST APIs for extraction and question answering.

---

# Tech Stack

- Python
- FastAPI
- Swagger UI
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

# Project Structure

```text
INVOICE_QA_INSIGHTs/

├── main.py
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
      charts/
      chroma_db/
      raw_invoices/
      clean_invoices.csv
      extracted_invoices.csv
```

---

# Workflow

```text
PDF Invoices
      │
      ▼
PDF Text Extraction (pdfplumber)
      │
      ▼
Qwen2.5 Local LLM
      │
      ├──────────────┐
      │              │
      ▼              ▼
JSON Validation    Retry
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
MiniLM Embeddings
             │
             ▼
ChromaDB Vector Database
             │
             ▼
Semantic Search (Top 4 Chunks)
             │
             ▼
Retrieval-Augmented Generation (RAG)
             │
             ▼
Question Answering
             │
             ▼
Statistics & Charts
```

---

# Features

- Local LLM using Qwen2.5-0.5B-Instruct
- Structured invoice extraction with Pydantic validation
- Automatic retry on validation failure
- Regex fallback extractor
- Prompt engineering with three extraction prompts
- Automatic prompt evaluation using manually labelled ground truth
- Semantic retrieval using ChromaDB and MiniLM embeddings
- Retrieval-Augmented Generation (RAG)
- Question answering with source citations
- Faithfulness check to reduce hallucinations
- Dataset cleaning and normalization
- Statistical analysis and visualization
- FastAPI REST API
- Interactive Swagger documentation

---

# API Endpoints

Start the API server:

```bash
uvicorn main:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

Available Endpoints:

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | / | Check API status |
| POST | /extract | Upload a PDF invoice and extract structured invoice data |
| POST | /ask | Ask questions about invoices using Retrieval-Augmented Generation (RAG) |

---

# How to Run

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Local LLM

```bash
python llm.py "Reply with the word PONG"
```

---

## Extract Invoice Data

```bash
python extract.py
```

---

## Build Retriever

```bash
python retriever.py
```

---

## Question Answering

```bash
python qa.py
```

---

## Evaluate Extraction Prompts

```bash
python eval/eval_extraction.py
```

---

## Evaluate Answer Prompts

```bash
python eval/eval_answers.py
```

---

## Test Model Limitations

```bash
python eval/test_limits.py
```

---

## Clean Dataset

```bash
python clean_data.py
```

---

## Generate Statistics and Charts

```bash
python analysis.py
```

---

## Run FastAPI

```bash
uvicorn main:app --reload
```

---

# Dataset

- 525 Invoice PDFs
- 20 manually labelled invoices for extraction evaluation
- 15 question-answer pairs for QA evaluation
- ChromaDB vector database

---

# Outputs

The project generates:

- extracted_invoices.csv
- clean_invoices.csv
- eval_extraction_scores.csv
- eval_answers_scores.csv
- failures_table.csv
- vendor_spend.png
- monthly_spend.png

---

# Results

- Successfully extracted structured invoice data using a local LLM.
- Evaluated three extraction prompt versions and selected the best-performing prompt.
- Built a semantic retriever using ChromaDB and MiniLM embeddings.
- Implemented Retrieval-Augmented Generation (RAG) for invoice question answering.
- Added a faithfulness check to reduce hallucinations.
- Cleaned and normalized invoice datasets.
- Generated statistical summaries and business insight charts.
- Exposed extraction and question answering through FastAPI with Swagger documentation.

---

# Future Improvements

- Improve retrieval accuracy using metadata-aware search.
- Support scanned invoices using OCR.
- Build a web interface using Streamlit or React.
- Deploy using Docker.
- Add authentication for API endpoints.
- Add unit tests.
- Add CI/CD pipeline.

---

# Author

**Renuka Avula**

AI/ML Internship Project – Invoice Q&A and Insights