# 🧾 Invoice Reimbursement System (AI/ML Internship Assignment)

## 🚀 Project Overview

This project is an **Intelligent Invoice Reimbursement System** that:
- Uses **AI (LLM)** to read and analyze invoices based on company reimbursement policy.
- Stores results in a **vector database** for smart search and querying.
- Provides a **chatbot API** for querying invoices using natural language.

---

## 🔧 Tech Stack

| Tool | Usage |
|------|-------|
| **Python** | Core Programming |
| **FastAPI** | REST APIs |
| **LangChain** | LLM tools |
| **OpenAI (or Groq)** | LLMs for invoice & chatbot |
| **Sentence-Transformers** | Text embedding |
| **ChromaDB** | Vector store |
| **PyMuPDF** | PDF text extraction |

---

## 📂 File Structure

```bash
Invoice_Reimbursement_Project/
├── main.py              # FastAPI entrypoint
├── invoice_processor.py # Handles invoice + policy processing
├── chatbot.py           # RAG chatbot to answer invoice queries
├── README.md            # Project documentation

