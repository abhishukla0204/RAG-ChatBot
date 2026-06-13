"""
app.py — Streamlit Chat UI for RAG Chatbot
============================================
A clean, chat-style interface to interact with the RAG pipeline.
Upload a PDF in the sidebar, process it, and ask questions!
"""

import os
import streamlit as st
from ingest import create_vectorstore, load_vectorstore, VECTORSTORE_DIR
from rag_engine import get_qa_chain

# ── Page Configuration ─────────────────────────────────────────
st.set_page_config(
    page_title="📚 RAG Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .main-header p {
        color: #888;
        font-size: 1.05rem;
    }

    /* Status badge */
    .status-badge {
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .status-ready {
        background: rgba(46, 213, 115, 0.15);
        color: #2ed573;
        border: 1px solid rgba(46, 213, 115, 0.3);
    }
    .status-waiting {
        background: rgba(255, 165, 2, 0.15);
        color: #ffa502;
        border: 1px solid rgba(255, 165, 2, 0.3);
    }

    /* Source documents expander */
    .source-chunk {
        background: rgba(102, 126, 234, 0.08);
        border-left: 3px solid #667eea;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Chat input */
    .stChatInput > div {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session State Initialization ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_ready" not in st.session_state:
    # Check if a vectorstore already exists on disk
    st.session_state.vectorstore_ready = os.path.exists(VECTORSTORE_DIR)
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# ── Sidebar: PDF Upload & Processing ──────────────────────────
with st.sidebar:
    st.markdown("## 📄 Document Upload")
    st.markdown("Upload a PDF to build the knowledge base.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload the document you want to ask questions about.",
    )

    if uploaded_file is not None:
        # Save uploaded PDF to data/ directory
        os.makedirs("data", exist_ok=True)
        pdf_path = os.path.join("data", uploaded_file.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"📎 **{uploaded_file.name}** uploaded!")

        # Process button
        if st.button("🔄 Process Document", use_container_width=True):
            with st.spinner("⏳ Processing... This may take a moment."):
                try:
                    create_vectorstore(pdf_path)
                    st.session_state.vectorstore_ready = True
                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.messages = []  # Clear chat for new doc
                    st.success("✅ Document processed! Start asking questions.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.markdown("---")

    # Status indicator
    if st.session_state.vectorstore_ready:
        status_text = f"✅ Ready"
        if st.session_state.processed_file:
            status_text += f" — {st.session_state.processed_file}"
        st.markdown(
            f'<div class="status-badge status-ready">{status_text}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-waiting">⏳ Upload a PDF to get started</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
        ### 🛠️ How it works
        1. **Upload** a PDF document
        2. **Process** it to build the index
        3. **Ask** questions about the content
        
        The chatbot answers **only** from your 
        document — no hallucination!
        """
    )

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Chat Interface ───────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🤖 RAG Chatbot</h1>
        <p>Ask questions about your document — powered by LangChain + FAISS + Gemini</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📄 View Source Chunks", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    page_num = source.metadata.get("page", "N/A")
                    st.markdown(
                        f'<div class="source-chunk">'
                        f"<strong>Chunk {i}</strong> (Page {page_num})<br>"
                        f"{source.page_content[:300]}..."
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    # Check if vectorstore is ready
    if not st.session_state.vectorstore_ready:
        st.warning("⚠️ Please upload and process a PDF first!")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document..."):
                try:
                    qa_chain = get_qa_chain()
                    result = qa_chain.invoke({"query": prompt})

                    answer = result["result"]
                    sources = result["source_documents"]

                    st.markdown(answer)

                    # Show source chunks
                    if sources:
                        with st.expander("📄 View Source Chunks", expanded=False):
                            for i, source in enumerate(sources, 1):
                                page_num = source.metadata.get("page", "N/A")
                                st.markdown(
                                    f'<div class="source-chunk">'
                                    f"<strong>Chunk {i}</strong> (Page {page_num})<br>"
                                    f"{source.page_content[:300]}..."
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    # Save to chat history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except ValueError as e:
                    st.error(str(e))
                except FileNotFoundError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"❌ Something went wrong: {e}")
