# 💻 Chat with Your Codebase (RAG)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Groq](https://img.shields.io/badge/LLM-Llama3.3-orange)

An intelligent **Retrieval-Augmented Generation (RAG)** application that allows developers to "chat" with their codebase. Upload your project files (Python, JS, Java, etc.) and ask questions about logic, dependencies, or bugs. The system uses **Groq** for ultra-fast inference and **FAISS** for local vector storage.

---

## 🚀 Key Features

* **📂 Multi-File Ingestion:** Drag & Drop support for `.py`, `.js`, `.java`, `.cpp`, `.md`, and more.
* **🧠 Smart Chunking:** Uses `RecursiveCharacterTextSplitter` (Python-aware) to split code by function/class boundaries rather than arbitrary lines.
* **⚡ Ultra-Fast Inference:** Powered by **Groq's LPU** (Language Processing Unit) using the **Llama 3.3 (70B)** model.
* **🔒 Privacy-First:** Uses local embeddings (**HuggingFace**) and a local vector store (**FAISS**). Your code vectors never leave your machine (except the snippet sent to Groq for the answer).
* **📝 Cited Answers:** The bot provides the answer *and* shows the exact source code snippets it referenced.
* **⚙️ Advanced Filters:** Adjust the number of retrieved chunks dynamically via the sidebar.

---

## 🛠️ Tech Stack

* **LLM:** Llama 3.3 (70B) via [Groq API](https://groq.com/)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
* **Vector DB:** FAISS (Facebook AI Similarity Search)
* **Orchestration:** LangChain
* **Frontend:** Streamlit

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/rag-codebase-chat.git](https://github.com/your-username/rag-codebase-chat.git)
cd rag-codebase-chat


2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

Bash

# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
4. Set Up Environment Keys
Create a .env file in the root directory:

Bash

touch .env # or create manually in VS Code
Add your Groq API Key (Get it from console.groq.com):

Code snippet

GROQ_API_KEY=gsk_your_actual_api_key_here
▶️ Usage
Run the Streamlit application:

Bash

streamlit run app.py
The app will open in your browser (usually http://localhost:8501).

Open the Sidebar on the left.

Upload your code files (e.g., select all .py files from your project).

Click "Process Files".

Once processing is complete, ask a question in the chat box!

Example Questions:

"Where is the authentication logic handled?"

"Explain the process_payment function step-by-step."

"Are there any hardcoded API keys in these files?"

📂 Project Structure
Plaintext

rag-codebase-chat/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # API keys (DO NOT COMMIT THIS)
├── ci_cd_pipeline.yaml     # (Optional) GitHub Actions workflow example
├── .gitignore              # Files to ignore in Git
└── README.md               # Documentation
🧪 Architecture
Ingestion: User uploads files -> TextLoader reads raw text.

Chunking: Code is split into manageable chunks (e.g., 1000 chars) using RecursiveCharacterTextSplitter.

Embedding: Chunks are converted into vectors using the all-MiniLM-L6-v2 model.

Storage: Vectors are stored locally in a FAISS index.

Retrieval: User question -> Vectorized -> Top-K similar chunks found in FAISS.

Generation: Retrieved chunks + Question sent to Groq Llama 3 -> Final Answer.

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request
