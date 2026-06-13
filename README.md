# 🤖 RAG Chatbot — Build a RAG Chatbot with LangChain

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions **only from your uploaded PDF document** — no hallucination. Built with LangChain, FAISS, HuggingFace Embeddings, and Google Gemini.

---

## 🏗️ Architecture

```
📄 PDF Document
    │
    ▼
┌──────────────┐
│  PyPDFLoader  │  ← Load PDF pages
└──────┬───────┘
       ▼
┌─────────────────────────────┐
│ RecursiveCharacterTextSplitter │  ← Chunk into 1000-char segments
└──────────┬──────────────────┘
           ▼
┌──────────────────────┐
│ HuggingFace Embeddings │  ← all-MiniLM-L6-v2 (local, free)
│ (sentence-transformers)│
└──────────┬───────────┘
           ▼
┌────────────────┐
│  FAISS Index   │  ← Vector store (saved locally)
└───────┬────────┘
        ▼
┌────────────────┐     ┌─────────────────┐
│   Retriever    │────▶│  RetrievalQA    │
│  (top-k = 4)  │     │    Chain        │
└────────────────┘     │                 │
                       │  + Google Gemini│
  👤 User Question ──▶ │  + Custom Prompt│ ──▶ 🤖 Grounded Answer
                       └─────────────────┘
```

## 📋 Pipeline Steps

| Step | Action | Tool |
|------|--------|------|
| 1. **Setup** | Install dependencies | `pip install -r requirements.txt` |
| 2. **Load** | Load PDF pages | `PyPDFLoader` |
| 3. **Split** | Chunk text into segments | `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap) |
| 4. **Embed & Store** | Generate embeddings & build index | `HuggingFaceEmbeddings` + `FAISS` |
| 5. **Retrieve + Generate** | Find relevant chunks & generate answer | `RetrievalQA` chain + `Gemini 2.0 Flash` |
| 6. **Test** | Ask questions, verify grounded answers | Streamlit UI |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd GenAITutorials

pip install -r requirements.txt
```

### 2. Set Up API Key

```bash
# Copy the template
cp .env.example .env

# Edit .env and add your Google Gemini API key
# Get a free key at: https://aistudio.google.com/apikey
```

### 3. Run the Chatbot

```bash
streamlit run app.py
```

### 4. Use the App

1. **Upload** a PDF file using the sidebar
2. Click **"Process Document"** to build the vector index
3. **Ask questions** in the chat input
4. View **source chunks** to verify answers are grounded

---

## 📁 Project Structure

```
GenAITutorials/
├── data/                 # Uploaded PDF files
├── vectorstore/          # Auto-generated FAISS index
├── app.py                # Streamlit UI (main entry point)
├── rag_engine.py         # RetrievalQA chain with Gemini
├── ingest.py             # PDF → Chunks → Embeddings → FAISS
├── requirements.txt      # Python dependencies
├── .env.example          # API key template
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

---

## 🛡️ Anti-Hallucination

The chatbot uses a **custom prompt** that strictly instructs the LLM to:
- Answer **only** from the provided document context
- Respond with **"Not found in document"** if the answer isn't available
- Never make up or infer information

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | LangChain |
| **Vector Store** | FAISS (local, CPU) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (local, free) |
| **LLM** | Google Gemini 2.0 Flash (free tier) |
| **PDF Loader** | PyPDFLoader |
| **Text Splitter** | RecursiveCharacterTextSplitter |
| **UI** | Streamlit |

---

## 💡 Usage Examples

**Question (in document):**  
> "What is the main topic of chapter 3?"  
> ✅ *"Chapter 3 discusses the fundamentals of neural networks, covering..."*

**Question (not in document):**  
> "What is the weather today?"  
> ❌ *"Not found in document."*

---

## 📝 License

This project is for educational purposes as part of the GenAI course assignment.
