from fastapi import FastAPI, UploadFile, Form
from invoice_processor import process_invoices
from chatbot import query_chatbot

app = FastAPI()

@app.post("/analyze-invoices/")
async def analyze(policy_pdf: UploadFile, invoice_zip: UploadFile, employee_name: str = Form(...)):
    return await process_invoices(policy_pdf, invoice_zip, employee_name)

@app.post("/chat/")
async def chat(query: str = Form(...)):
    return query_chatbot(query)
