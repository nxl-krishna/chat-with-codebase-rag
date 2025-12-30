import streamlit as st
import os
import shutil
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_classic.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Chat with Codebase", layout="wide")

# --- PATH CONFIGURATION ---
WORKING_DIR = os.getcwd()
UPLOAD_DIR = os.path.join(WORKING_DIR, "uploaded_code")
DB_FAISS_PATH = os.path.join(WORKING_DIR, "vectorstore", "db_faiss")

# Ensure directories exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- BACKEND FUNCTIONS ---

@st.cache_resource
def get_embedding_model():
    # Load the HuggingFace model once (Cached)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_uploaded_files(uploaded_files):
    # 1. Clear previous files to avoid mixing projects
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)

    # 2. Save new files
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # 3. Load & Split
    # loader_cls=TextLoader ensures we treat everything as plain text code
    loader = DirectoryLoader(UPLOAD_DIR, glob="**/*", loader_cls=TextLoader)
    documents = loader.load()
    
    # Text splitter for code (chunk_size=1000 is good for functions)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 4. Create Vector DB
    embeddings = get_embedding_model()
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db

def get_qa_chain(vector_store):
    # Using the Llama 3.3 model from Groq
    llm = ChatGroq(
        groq_api_key=os.environ['GROQ_API_KEY'],
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )
    
    # Prompt template to keep answers grounded in the code
    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

# --- FRONTEND UI ---

st.title("💻 Chat with Your Codebase")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📁 Project Files")
    st.write("Upload your code files to start chatting.")
    
    uploaded_files = st.file_uploader(
        "Upload Code Files", 
        accept_multiple_files=True, 
        type=['py', 'js', 'txt', 'md', 'java', 'cpp', 'c']
    )
    
    if st.button("Process Files"):
        if uploaded_files:
            with st.spinner("Analyzing codebase..."):
                vector_db = process_uploaded_files(uploaded_files)
                st.session_state.vector_store = vector_db
                st.success("Codebase processed! You can now chat.")
        else:
            st.warning("Please upload files first.")
            
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT ---

if st.session_state.vector_store is None:
    st.info("👈 Please upload your code files in the sidebar to begin.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your code..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                chain = get_qa_chain(st.session_state.vector_store)
                response = chain.invoke({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]
                
                message_placeholder.markdown(answer)
                
                with st.expander("View Source Code Segments"):
                    for i, doc in enumerate(sources):
                        source_name = os.path.basename(doc.metadata['source'])
                        st.markdown(f"**Source {i+1}:** `{source_name}`")
                        st.code(doc.page_content)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                message_placeholder.error(f"Error: {str(e)}")