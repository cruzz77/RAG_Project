import asyncio
from pathlib import Path
import time
import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests
from chat_history import ChatHistoryManager  # 🆕 NEW

load_dotenv()

@st.cache_resource
def get_chat_manager():
    return ChatHistoryManager()

chat_manager = get_chat_manager()

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="wide",  
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message-user {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #2196f3;
    }
    .chat-message-bot {
        background-color:  #e3f2fd;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #9c27b0;
    }
    .history-item {
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .history-item:hover {
        background-color: #f0f0f0;
    }
    .history-item.active {
        background-color: #1f77b4;
        color: white;
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
                "top_k": 5,
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

if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'current_pdf_name' not in st.session_state:
    st.session_state.current_pdf_name = None

# Chat History sidebar
with st.sidebar:
    st.markdown("## 📚 Chat History")
    
    sessions = chat_manager.get_all_sessions()
    if sessions:
        st.markdown("### Previous Chats")
        for session in sessions:
            is_active = session.session_id == st.session_state.current_session_id
            emoji = "🔵" if is_active else "⚪"
            if st.button(
                f"{emoji} {session.pdf_name} ({len(session.messages)} messages)",
                key=f"session_{session.session_id}",
                use_container_width=True
            ):
                st.session_state.current_session_id = session.session_id
                st.session_state.current_pdf_name = session.pdf_name
                st.rerun()
    
    st.markdown("---")
    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.current_pdf_name = None
        st.rerun()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">📚 RAG PDF Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Your Intelligent Document Analysis Tool")

st.markdown("---")

if st.session_state.current_session_id and st.session_state.current_pdf_name:
    st.success(f"💬 Currently chatting about: **{st.session_state.current_pdf_name}**")

st.markdown("### 📤 Upload Your PDF Document")
with st.container():
    uploaded = st.file_uploader(
        "Drag and drop or click to upload PDF", 
        type=["pdf"], 
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

if uploaded is not None:
    with st.spinner("🔄 Uploading and processing your document..."):
        path = save_uploaded_pdf(uploaded)
        send_rag_ingest_event_sync(path)
        
        if not st.session_state.current_session_id:
            session_id = chat_manager.create_session(path.name)
            st.session_state.current_session_id = session_id
            st.session_state.current_pdf_name = path.name
        
        time.sleep(0.3)
    
    st.success(f"✅ Document **{path.name}** uploaded and ready for questions!")

st.markdown("---")

if st.session_state.current_session_id:
    messages = chat_manager.get_session_history(st.session_state.current_session_id)
    
    if messages:
        st.markdown("### 💬 Conversation History")
        for msg in messages:
            st.markdown(f'<div class="chat-message-user">👤 **You:** {msg.question}</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="chat-message-bot">🤖 **Assistant:** {msg.answer}</div>', unsafe_allow_html=True)
            
            with st.expander(f"📚 Sources ({len(msg.sources)})"):
                for source in msg.sources:
                    st.write(f"• {source}")

st.markdown("### 💭 Ask a Question")
with st.form("rag_query_form", clear_on_submit=False):
    question = st.text_area(
        "Enter your question:",
        placeholder="e.g., What are the main points of this document?",
        height=100,
        key="question_input"
    )
    
    submitted = st.form_submit_button("🚀 Get Answer", use_container_width=True)

    if submitted and question.strip():
        if not st.session_state.current_session_id:
            session_id = chat_manager.create_session("General Chat")
            st.session_state.current_session_id = session_id
            st.session_state.current_pdf_name = "General Chat"
        
        with st.spinner("🔍 Searching document and generating answer..."):
            event_id = send_rag_query_event_sync(question.strip())
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])
            num_contexts = output.get("num_contexts", 0)

        if answer:
            chat_manager.add_message(
                st.session_state.current_session_id,
                question.strip(),
                answer,
                sources
            )
            st.rerun()  

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; margin-top: 50px;'>
    <p>Built with ❤️ using Streamlit, FastAPI, and RAG Technology</p>
    <p>Powered by Sentence Transformers & Groq AI</p>
</div>
""", unsafe_allow_html=True)
