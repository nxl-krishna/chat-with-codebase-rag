import streamlit as st
import os
import shutil
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_classic.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Chat with Codebase", layout="wide")

# --- CONFIG ---
WORKING_DIR = os.getcwd()
UPLOAD_DIR = os.path.join(WORKING_DIR, "uploaded_code")
DB_FAISS_PATH = os.path.join(WORKING_DIR, "vectorstore", "db_faiss")

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- BACKEND ---
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_uploaded_files(uploaded_files):
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)

    documents = []
    
    # 1. Save and Load Files
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Load immediately with metadata
        loader = TextLoader(file_path)
        docs = loader.load()
        # Add simpler filename metadata for the UI
        for doc in docs:
            doc.metadata["file_name"] = uploaded_file.name
            documents.extend(docs)

    # 2. Smart Splitting (The "Code Parsing" Requirement)
    # We use .from_language to respect Python syntax (classes/functions)
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, 
        chunk_size=1000, 
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)

    # 3. Create Embeddings
    embeddings = get_embedding_model()
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db

def get_qa_chain(vector_store, num_chunks):
    llm = ChatGroq(
        groq_api_key=os.environ['GROQ_API_KEY'],
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )
    
    prompt_template = """Use the following pieces of context to answer the question.
    If you don't know, say "I don't know".
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": num_chunks}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

# --- FRONTEND ---
st.title("💻 Chat with Your Codebase")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.header("📁 Project Files")
    uploaded_files = st.file_uploader("Upload Code", accept_multiple_files=True, type=['py', 'js', 'java', 'md'])
    
    if st.button("Process Files"):
        if uploaded_files:
            with st.spinner("Parsing code syntax & creating index..."):
                st.session_state.vector_store = process_uploaded_files(uploaded_files)
                st.success("Analysis Complete!")
        else:
            st.warning("Upload files first.")

    st.markdown("---")
    st.header("⚙️ Advanced Filters")
    # FEATURE: Metadata Filtering (from Slide)
    k_val = st.slider("Chunks to retrieve:", 1, 10, 3)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if st.session_state.vector_store is None:
    st.info("👈 Upload code in the sidebar to begin.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about logic, functions, or dependencies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Pass the slider value to the chain
                chain = get_qa_chain(st.session_state.vector_store, k_val)
                response = chain.invoke({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]
                
                message_placeholder.markdown(answer)
                
                with st.expander("View Source Code Segments"):
                    for i, doc in enumerate(sources):
                        # Clean up the path to just show filename
                        name = doc.metadata.get('file_name', 'unknown')
                        st.markdown(f"**Reference {i+1}:** `{name}`")
                        st.code(doc.page_content, language="python")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                message_placeholder.error(f"Error: {str(e)}")