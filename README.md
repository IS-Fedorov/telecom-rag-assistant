# Customer Support RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system designed to answer customer questions about mobile operator tariffs, options, and services based on a structured knowledge base.

## Overview
Standard LLMs often hallucinate specific tariff conditions or pricing. This project implements a RAG pipeline that retrieves relevant tariff documentation and generates accurate, grounded answers, validated through an automated 55-question evaluation benchmark.

## Tech Stack
* **Language:** Python
* **RAG Framework:** LangChain
* **Vector Store & Retrieval:** FAISS, BM25 (Hybrid Search), Multilingual Embeddings
* **LLM & Inference:** Qwen 2.5 (via OpenRouter API)
* **Evaluation & Analysis:** Pandas, Jupyter Notebook

## Architecture & Workflow
1. **Document Processing & Chunking:** Splitting tariff descriptions into semantic chunks (tested 250 and 500 character windows with overlap).
2. **Indexing:** Encoding chunks with multilingual embeddings into a FAISS vector index.
3. **Retrieval:** Fetching the top-k most relevant context passages (`top_k=9`) using dense vector search and hybrid keyword-vector retrieval (BM25 + FAISS).
4. **Generation:** Passing retrieved context and user query to Qwen 2.5 with a strict system prompt to prevent hallucinations.
5. **Evaluation:** Automated benchmarking against 55 ground-truth Q&A pairs.

## Experiments & Results
To optimize retrieval quality and response accuracy, 7 iterative experiments were conducted:

| Stage / Configuration | Key Changes | Accuracy (55 Q&A) |
| :--- | :--- | :---: |
| **Baseline** | Default chunking, basic embeddings, `top_k=3` | **40.00%** |
| **Peak Configuration** | Switched to multilingual embeddings + expanded context (`top_k=9`) | **90.91%** |

## Repository Structure
```text
├── knowledge_base.txt             # Telecom tariff knowledge base
├── eval_dataset.csv               # 55-question ground-truth evaluation benchmark
├── rag_pipeline_experiments.ipynb # Main RAG pipeline & iterative experiments
├── baseline_results.csv           # Baseline evaluation logs (40.00% accuracy)
├── best_results.csv               # Final pipeline evaluation logs (90.91% accuracy)
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation