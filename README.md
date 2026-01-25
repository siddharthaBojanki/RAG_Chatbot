# 🔭 Orion's Gaze

**A Streamlit-powered RAG Chatbot that lets you chat with your PDFs using Gemini + LangChain.**  
Upload documents, ask questions, and get context-aware answers — powered by **vector search**, **embeddings**, and **Google Gemini 2.5**.

---

## 🚀 Overview

**Orion's Gaze** is an intelligent document analysis and Q&A assistant that uses **Retrieval-Augmented Generation (RAG)** to combine the power of **local embeddings** with **LLM reasoning**.

It allows users to:
- Upload **PDF documents**.
- Ask **contextual questions** related to uploaded files.
- Get accurate, source-backed answers powered by **Google Gemini**.
- Optionally allow **LLM fallback mode** (combine document + general model knowledge).

---

## ✨ Features

✅ **Multiple PDF Uploads** – Process and index several PDFs at once.  
✅ **Vector-Based Retrieval** – Uses **Hugging Face Embeddings** + **LangChain FAISS** for efficient context search.  
✅ **Streamed Responses** – See Gemini’s responses type out in real time.  
✅ **Context Preview** – View chunks retrieved from your documents.  
✅ **LLM Fallback Mode** – Hybrid reasoning (document + model knowledge).  
✅ **Elegant UI** – Built using **Streamlit** with custom styling.  
✅ **Persistent Chat History** – Maintains messages across interactions.  

---

## 🧠 Architecture

**RAG Pipeline:**
1. Upload PDF → Extract & Split text → Embed via `thenlper/gte-small`.
2. Store embeddings in FAISS vector store.
3. On query:
   - Retrieve top relevant chunks.
   - Pass context + question to Gemini model.
4. Stream generated answer in real time.

---

## 🛠️ Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| Backend | Python |
| Embeddings | HuggingFace (`thenlper/gte-small`) |
| Vector Store | LangChain FAISS |
| LLM | Google Gemini 2.5 Flash (`ChatGoogleGenerativeAI`) |
| Environment Management | Python-dotenv |
| File Handling | PyPDFLoader, Tempfile |
| Styling | Custom Streamlit CSS |

---

## 📦 Installation

```bash
# 1️⃣ Clone the repository
https://github.com/siddharthaBojanki/RAG_Chatbot.git
cd Orions_Gaze

# 2️⃣ Create a virtual environment
python -m venv venv
venv\Scripts\activate     # (on Windows)
# source venv/bin/activate   (on Linux/Mac)

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Add your Gemini API key in a .env file
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# 5️⃣ Run the Streamlit app
streamlit run Orion_gaze.py
