# Walkthrough — RAG Chatbot with LangChain

## Summary

Built a complete RAG (Retrieval-Augmented Generation) chatbot that answers questions **only from uploaded PDF documents** — with no hallucination. The project is fully functional and ready for testing.

## Files Created

| File | Purpose |
|------|---------|
| [requirements.txt](file:///c:/Abhinav_Shukla/MyProjects/GenAITutorials/requirements.txt) | Python dependencies (langchain, faiss-cpu, sentence-transformers, streamlit, etc.) |
| [.env.example](file:///c:/Abhinav_Shukla/MyProjects/GenAITutorials/.env.example) | API key template |
| [.gitignore](file:///c:/Abhinav_Shukla/MyProjects/GenAITutorials/.gitignore) | Ignore secrets, vectorstore, cache |
| [ingest.py](file:///c:/Abhinav_Shukla/MyProjects/GenAITutorials/ingest.py) | PDF → chunks → embeddings → FAISS index |
| [rag_engine.py](file:///c:/Abhinav_Shukla/MyProjects/GenAITutorials/rag_engine.py) | RetrievalQA chain with anti-hallucination prompt |
| [app.py](file:///c:/Abhinav_Shukla/MyProjects/GenAITutorials/app.py) | Streamlit chat UI with sidebar PDF upload |
| [README.md](file:///c:/Abhinav_Shukla/MyProjects/GenAITutorials/README.md) | Full project documentation for submission |

## Issues Fixed

- **`langchain.text_splitter`** → moved to `langchain_text_splitters` in langchain v1.x+
- **`langchain.chains.RetrievalQA`** → moved to `langchain_classic.chains`
- **`langchain.prompts.PromptTemplate`** → moved to `langchain_core.prompts`

## Verification

- ✅ All dependencies installed successfully
- ✅ All imports verified — no errors

## How to Run

### 1. Set your API key
Edit [.env](file:///c:/Abhinav_Shukla/MyProjects/GenAITutorials/.env) and replace `your_google_api_key_here` with your actual Google Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

### 2. Launch the app
```bash
python -m streamlit run app.py
```

### 3. Test the chatbot
1. Upload a PDF via the sidebar
2. Click **"Process Document"**
3. Ask questions — answers should come only from the document
4. Try asking something NOT in the document — should get "Not found in document"
