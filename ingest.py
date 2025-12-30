import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
REPO_PATH = "./my_codebase"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    print("--- Step 1: Loading Documents ---")
    loader = DirectoryLoader(REPO_PATH, glob="./**/*.py", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    print("--- Step 2: Splitting Text into Chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks.")

    print("--- Step 3: Creating Embeddings (HuggingFace) ---")
    # This will download a small model (~80MB) the first time you run it.
    # It runs locally on your machine.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("--- Step 4: Saving to FAISS ---")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print("--- Vector Database Saved Successfully ---")

if __name__ == "__main__":
    create_vector_db()