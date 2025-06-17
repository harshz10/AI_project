import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq


# Load embedding model & vector DB
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
VECTOR_DIR = "vector_db"
vector_db = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding_model)

# Setup retriever
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Setup LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",
    api_key=os.getenv("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")
)

# Load retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Function to answer query
def query_chatbot(query: str):
    response = qa_chain.run(query)
    return {"response": response}
