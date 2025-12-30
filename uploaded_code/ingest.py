import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load PDF
def load_pdf(file_path):
    print(f"Loading PDF: {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# 2. Load Website
def load_website(url):
    print(f"Loading Website: {url}...")
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents

# 3. Load CSV
def load_csv(file_path):
    print(f"Loading CSV: {file_path}...")
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    return documents

# 4. Split Text (Chunking)
# We split huge documents into smaller chunks so the AI can process them.
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# Test the ingestion (Only runs if you execute this file directly)
if __name__ == "__main__":
    # create a dummy 'data' folder and put a sample PDF and CSV inside it to test
    # Or just comment out what you don't have yet.
    
    # Example usage:
    pdf_docs = load_pdf("data/sample.pdf")
    web_docs = load_website("https://en.wikipedia.org/wiki/Artificial_intelligence")
    # csv_docs = load_csv("data/sample.csv")
    
    combined_docs = pdf_docs + web_docs # + csv_docs
    chunks = split_documents(combined_docs)
    print(chunks[0].page_content)
    pass