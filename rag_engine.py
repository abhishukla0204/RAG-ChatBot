"""
rag_engine.py — RAG Q&A Engine
================================
Wires the FAISS retriever into a LangChain RetrievalQA chain
with Google Gemini as the LLM. Uses a custom prompt to prevent
hallucination — the model only answers from the document context.
"""

import os
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from ingest import load_vectorstore

# Load environment variables (.env file)
load_dotenv()

# ── Anti-Hallucination Prompt ──────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions 
based ONLY on the provided document context. 

Rules:
1. Answer the question using ONLY the information in the context below.
2. If the answer is NOT found in the context, respond with exactly: 
   "❌ Not found in document."
3. Do NOT make up or infer information beyond what is explicitly stated.
4. Be concise and accurate in your responses.
5. If relevant, mention which part of the document the answer comes from.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def get_llm():
    """Initialize Google Gemini LLM (free tier)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        raise ValueError(
            "⚠️ GOOGLE_API_KEY not set! "
            "Please add your key to the .env file.\n"
            "Get a free key at: https://aistudio.google.com/apikey"
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.2,  # Low temperature for factual answers
        convert_system_message_to_human=True,
    )


def get_qa_chain():
    """
    Build and return the complete RetrievalQA chain.

    Pipeline:
    1. Load FAISS vectorstore from disk
    2. Create a retriever (top-k=4 most relevant chunks)
    3. Wire into RetrievalQA with custom anti-hallucination prompt
    4. Use Google Gemini as the LLM
    """
    # Load vectorstore
    vectorstore = load_vectorstore()

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},  # Return top 4 most relevant chunks
    )

    # Build QA chain
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Stuff all retrieved docs into the prompt
        retriever=retriever,
        return_source_documents=True,  # Return source chunks for transparency
        chain_type_kwargs={"prompt": PROMPT},
    )

    return qa_chain


def ask_question(question: str) -> dict:
    """
    Ask a question and get a grounded answer.

    Returns:
        dict with 'answer' and 'source_documents'
    """
    qa_chain = get_qa_chain()
    result = qa_chain.invoke({"query": question})

    return {
        "answer": result["result"],
        "source_documents": result["source_documents"],
    }


# ── Standalone usage ───────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 RAG Chatbot — Type 'quit' to exit\n")

    while True:
        question = input("❓ Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if not question:
            continue

        try:
            result = ask_question(question)
            print(f"\n💡 Answer: {result['answer']}\n")
            print(f"📄 Sources: {len(result['source_documents'])} chunk(s) used")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Error: {e}\n")
