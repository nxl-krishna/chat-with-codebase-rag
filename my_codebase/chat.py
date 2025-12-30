import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_classic.prompts import PromptTemplate

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

def run_chat_bot():
    # 1. Load the Vector Database (The Knowledge)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    # 2. Setup the Retriever
    # 'k=3' means we will fetch the top 3 most similar code chunks
    retriever = db.as_retriever(search_kwargs={'k': 3})

    # 3. Setup the Prompt Template (Instructions for the LLM)
    custom_prompt_template = """Use the following pieces of code context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    Question: {question}
    
    Only return the helpful answer below and mentions the file names if available:
    """
    
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    # 4. Load the LLM (Gemini Flash as recommended in your image)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # 5. Create the Retrieval Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "Stuff" puts all chunks into the prompt at once
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    # 6. Interactive Loop
    while True:
        query = input("\nAsk a question about your code (or type 'exit'): ")
        if query.lower() == 'exit':
            break
            
        # Get result
        result = qa_chain.invoke({"query": query})
        print(f"\nAnswer: {result['result']}")
        
        # Show which files were used (Citations)
        print("\nSources used:")
        for doc in result['source_documents']:
            print(f"- {doc.metadata['source']}")

if __name__ == "__main__":
    run_chat_bot()