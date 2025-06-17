import os
import zipfile
import shutil
from dotenv import load_dotenv
load_dotenv()
import fitz  # PyMuPDF
from fastapi import UploadFile
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage

import uuid

# Setup LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",
    api_key=os.getenv("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")
)


# Setup Embeddings + VectorStore
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
VECTOR_DIR = "vector_db"
if not os.path.exists(VECTOR_DIR):
    os.makedirs(VECTOR_DIR)

db = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding_model)

# Function to read PDF text


def extract_text_from_pdf(uploaded_pdf: UploadFile):
    temp_filename = f"temp_{uuid.uuid4().hex}.pdf"
    with open(temp_filename, "wb") as f:
        content = uploaded_pdf.file.read()
        f.write(content)

    doc = fitz.open(temp_filename)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    os.remove(temp_filename)
    return text


# Function to unzip and extract invoice PDFs
def extract_invoices_from_zip(zip_file: UploadFile):
    zip_path = "invoices.zip"
    with open(zip_path, "wb") as f:
        f.write(zip_file.file.read())

    extract_dir = "invoices"
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(zip_path)
    return extract_dir

# MAIN: Process invoices
async def process_invoices(policy_pdf: UploadFile, invoice_zip: UploadFile, employee_name: str):
    print("🔍 Step 1: Extracting policy text...")
    policy_text = extract_text_from_pdf(policy_pdf)

    print("📦 Step 2: Extracting invoice files from ZIP...")
    invoice_dir = extract_invoices_from_zip(invoice_zip)

    results = []

    for invoice_filename in os.listdir(invoice_dir):
        print(f"🧾 Processing invoice: {invoice_filename}")
        if not invoice_filename.endswith(".pdf"):
            continue

        invoice_path = os.path.join(invoice_dir, invoice_filename)
        with open(invoice_path, "rb") as f:
            invoice_file = UploadFile(filename=invoice_filename, file=f)
            invoice_text = extract_text_from_pdf(invoice_file)

        print("🤖 Asking LLM for reimbursement decision...")

        prompt = f"""
You are a reimbursement expert. Check the below invoice against the company policy.

### Policy:
{policy_text}

### Invoice:
{invoice_text}

Tell if the invoice is: Fully Reimbursed, Partially Reimbursed, or Declined.
Also explain the reason.
        """

        response = llm([HumanMessage(content=prompt)])
        print("✅ LLM response received.")

        metadata = {
            "employee": employee_name,
            "invoice": invoice_filename,
        }

        print("💾 Saving to vector DB...")
        db.add_texts([invoice_text + "\n\n" + response.content], metadatas=[metadata])

        results.append({
            "invoice": invoice_filename,
            "analysis": response
        })

    shutil.rmtree(invoice_dir)

    print("🎉 All invoices processed successfully.")
    return {"success": True, "results": results}
