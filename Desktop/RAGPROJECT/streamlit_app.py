import asyncio
from pathlib import Path
import time
import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests
import base64

load_dotenv()

# 🎨 Enhanced Page Configuration
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 🎨 Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .success-box {
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .upload-box {
        background-color: #f1e8fd;
        border: 2px dashed #1f77b4;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .answer-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .source-item {
        background-color: #e9ecef;
        border-radius: 8px;
        padding: 10px 15px;
        margin: 5px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)

def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path

def send_rag_ingest_event_sync(pdf_path: Path) -> None:
    client = get_inngest_client()
    client.send_sync(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )

def send_rag_query_event_sync(question: str) -> str:
    client = get_inngest_client()
    result = client.send_sync(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": 5,  # 🎯 Fixed to 5 chunks - removed user option
            },
        )
    )
    return result[0] if result else ""

def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])

def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)

# 🎨 Header Section with Logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">📚 RAG PDF Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Your Intelligent Document Analysis Tool")

st.markdown("---")

# 📄 PDF Upload Section
st.markdown('<div class="sub-header">📤 Upload Your PDF Document</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag and drop or click to upload PDF", 
        type=["pdf"], 
        accept_multiple_files=False,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded is not None:
    with st.spinner("🔄 Uploading and processing your document..."):
        path = save_uploaded_pdf(uploaded)
        send_rag_ingest_event_sync(path)
        time.sleep(0.3)
    
    st.markdown(f"""
    <div class="success-box">
        <h4>✅ Success!</h4>
        <p>Document <strong>{path.name}</strong> has been uploaded and is being processed.</p>
        <p><em>You can now ask questions about this document below.</em></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 💬 Question Section
st.markdown('<div class="sub-header">💬 Ask Questions About Your Document</div>', unsafe_allow_html=True)

with st.form("rag_query_form", clear_on_submit=True):
    question = st.text_area(
        "Enter your question here:",
        placeholder="e.g., What are the main points of this document? Summarize the key findings...",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submitted = st.form_submit_button(
            "🚀 Get Answer", 
            use_container_width=True,
            type="primary"
        )

    if submitted and question.strip():
        with st.spinner("🔍 Searching document and generating answer..."):
            event_id = send_rag_query_event_sync(question.strip())
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])
            num_contexts = output.get("num_contexts", 0)

        # 🎨 Answer Display
        st.markdown("---")
        st.markdown("### 📝 Answer")
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        if answer:
            st.markdown(answer)
        else:
            st.info("No answer could be generated from the document.")
        st.markdown('</div>', unsafe_allow_html=True)

        # 🎨 Sources Display
        if sources:
            st.markdown("### 📚 Sources")
            st.caption(f"Found in {num_contexts} relevant sections")
            for source in sources:
                st.markdown(f'<div class="source-item">📄 {source}</div>', unsafe_allow_html=True)

# 🎨 Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; margin-top: 50px;'>
    <p>Built with ❤️ using Streamlit, FastAPI, and RAG Technology</p>
    <p>Powered by Sentence Transformers & Groq AI</p>
</div>
""", unsafe_allow_html=True)