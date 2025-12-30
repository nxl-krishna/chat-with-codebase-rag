import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_classic.prompts import PromptTemplate

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

def run_chat_bot():
    # 1. Load the Local Embeddings
    print("Loading local embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Load the Database
    # allow_dangerous_deserialization is needed because we are loading a pickle file
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={'k': 3})

    # 3. Setup Prompt
    custom_prompt_template = """Use the following pieces of code context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    Question: {question}
    
    Only return the helpful answer below and mentions the file names if available:
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    # 4. Load Groq (Updated Model Name)
    print("Loading Groq Llama 3.3 model...")
    llm = ChatGroq(
        groq_api_key=os.environ['GROQ_API_KEY'],
        model_name="llama-3.3-70b-versatile", 
        temperature=0.2
    )

    # 5. Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    print("\nBot is ready! (Using Local Embeddings + Groq Llama 3.3)")
    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == 'exit':
            break
            
        result = qa_chain.invoke({"query": query})
        print(f"\nAnswer: {result['result']}")

if __name__ == "__main__":
    run_chat_bot()