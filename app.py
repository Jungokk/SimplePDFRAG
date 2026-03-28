import streamlit as st
import time
import os
import json
import uuid
from datetime import datetime

from retrieval_module import BM25Retriever, FAISSRetriever, load_pdf_chunks
from generation_module import QwenRAGGenerator, AgenticRAGSystem

st.set_page_config(page_title="Agentic RAG Search", layout="wide", page_icon="🔍")


st.markdown("""
<style>
    .stApp {
        background-color: #F8F9FA !important;
        color: #1E293B !important;
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit chrome; keep sidebar toggle button visible */
    header {background: transparent !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Landing page */
    .landing-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 80px;
        margin-bottom: 10px;
    }
    .landing-subtitle {
        text-align: center;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 40px;
    }

    /* Top navigation bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        padding: 15px 20px;
        background: #FFFFFF;
        border-bottom: 1px solid #E2E8F0;
        align-items: center;
        margin: -4rem -4rem 2rem -4rem; /* 抵消 streamlit padding */
    }

    /* Thinking progress container */
    .thinking-container {
        background-color: #FFFFFF;
        padding: 20px 30px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
        border: 1px solid #E2E8F0;
    }
    .step-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 20px;
        position: relative;
    }
    .step-line {
        position: absolute;
        top: 20px;
        left: 40px;
        right: 40px;
        height: 2px;
        background-color: #E2E8F0;
        z-index: 0;
    }
    .step-item {
        position: relative;
        z-index: 1;
        text-align: center;
        width: 120px;
    }
    .step-circle {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #F1F5F9;
        color: #94A3B8;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 10px auto;
        font-weight: bold;
        transition: all 0.3s ease;
        border: 3px solid #FFFFFF;
    }
    /* Active step */
    .step-active .step-circle {
        background-color: #6366F1;
        color: white;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }
    .step-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
    }
    .step-active .step-label {
        color: #1E293B;
        font-weight: 700;
    }

    /* Retrieval cards */
    .card-container {
        display: flex;
        gap: 15px;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }
    .rag-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        flex: 1;
        min-width: 280px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        transition: transform 0.2s;
    }
    .rag-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px -3px rgba(0, 0, 0, 0.05);
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .match-score {
        background-color: #DCFCE7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 10px;
    }
    .card-title {
        font-weight: 700;
        color: #334155;
        margin-bottom: 6px;
        font-size: 0.95rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .card-content {
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.5;
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    /* Final answer box */
    .final-answer-box {
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        color: #334155;
        line-height: 1.6;
    }

    /* Chat input styling */
    div[data-testid="stTextInput"] input {
        border-radius: 24px !important;
        padding: 12px 20px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── History ──────────────────────────────
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return {}
    return {}

def save_history(data):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if not os.path.exists(HISTORY_FILE):
    save_history({"demo": []})

all_histories = load_history()

# ── Sidebar ───────────────────────────────
with st.sidebar:
    st.markdown("### 🗂️ History")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### 📂 PDF Knowledge Base")
    uploaded_pdfs = st.file_uploader(
        "Upload PDF files", type="pdf",
        accept_multiple_files=True,
        help="Upload PDFs to use as the retrieval corpus"
    )
    if uploaded_pdfs:
        if st.button("🔄 Build Index from PDFs", use_container_width=True):
            import tempfile
            pdf_paths = []
            for uploaded in uploaded_pdfs:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    pdf_paths.append(tmp.name)
            with st.spinner("Extracting PDF text and building FAISS index..."):
                pdf_collection = load_pdf_chunks(pdf_paths)
                if pdf_collection:
                    st.session_state.pdf_collection = pdf_collection
                    st.session_state.pdf_retriever = FAISSRetriever(pdf_collection)
                    st.success(f"✅ Indexed {len(pdf_collection)} chunks from {len(uploaded_pdfs)} PDF(s)")
                else:
                    st.error("No text extracted from PDFs.")

    if "pdf_retriever" in st.session_state:
        st.info(f"📑 PDF index active ({len(st.session_state.pdf_collection)} chunks)")
        if st.button("🗑️ Clear PDF Index", use_container_width=True):
            del st.session_state.pdf_collection
            del st.session_state.pdf_retriever
            st.rerun()

    st.divider()

    valid_sessions = {k: v for k, v in all_histories.items() if len(v) > 0}
    for s_id, msgs in sorted(valid_sessions.items(), key=lambda x: x[1][-1].get("timestamp", ""), reverse=True):
        title = msgs[0]["content"][:15] + "..."
        if st.button(f"📄 {title}", key=s_id, use_container_width=True):
            st.session_state.session_id = s_id
            st.rerun()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

current_id = st.session_state.session_id
if current_id in all_histories:
    st.session_state.messages = all_histories[current_id]
else:
    st.session_state.messages = []

# ── System initialization ─────────────────
COLAB_PATH = "/content/drive/MyDrive/Colab Notebooks/DSAI5201/corpus.jsonl"
LOCAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.jsonl")
CORPUS_PATH = COLAB_PATH if os.path.exists("/content") else LOCAL_PATH

@st.cache_resource
def initialize_system():
    if not os.path.exists(CORPUS_PATH): return None, []
    collection = []
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try: collection.append(json.loads(line))
            except: continue
    retriever = BM25Retriever(collection)
    generator = QwenRAGGenerator(model_name="Qwen/Qwen3-0.6B")
    rag_system = AgenticRAGSystem(collection, retriever, generator)
    return rag_system, collection

try:
    agent_system, doc_collection = initialize_system()
except Exception as e:
    st.error(f"System initialization failed: {e}")
    st.stop()

# Use PDF FAISS index if uploaded, otherwise fall back to BM25 on corpus.jsonl
if "pdf_collection" in st.session_state and "pdf_retriever" in st.session_state:
    doc_collection = st.session_state.pdf_collection
    if agent_system:
        agent_system.collection = doc_collection
        agent_system.retriever = st.session_state.pdf_retriever

if agent_system is None:
    st.warning("⚠️ No corpus loaded. Please upload PDF files in the sidebar.")
    st.stop()


# ── Chat interface ────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("thoughts"):
            with st.expander("💭 Reasoning Process", expanded=False):
                st.markdown(msg["thoughts"], unsafe_allow_html=True)
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a complex question..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("🚀 Starting Agentic Workflow...", expanded=True)
        thought_buffer = ""

        try:
            # Step 1: Decomposition
            status.write("📝 **Step 1: Decomposing Question...**")
            sub_queries = agent_system._decompose_complex_query(prompt)
            status.markdown(f"> *Sub-queries:* `{sub_queries}`")
            thought_buffer += f"### 📝 Decomposition\nSub-queries: {sub_queries}\n\n"

            # Step 2: Retrieval
            status.write(f"🔍 **Step 2: Searching Evidence for {len(sub_queries)} queries...**")

            if len(sub_queries) > 1:
                raw_results = agent_system._parallel_retrieval(sub_queries, k=2)
                best_scores = {}
                for q_res in raw_results.values():
                    for doc_id, score in q_res:
                        if doc_id not in best_scores or score > best_scores[doc_id]:
                            best_scores[doc_id] = score
                retrieved_items = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            else:
                retrieved_items = agent_system.retriever.retrieve(prompt, k=3)

            # build HTML using the CSS classes defined above so styling applies
            cards_html = "<div class='card-container'>"
            for d in retrieved_items:
                txt = next((i['text'] for i in doc_collection if i['id'] == d[0]), "")[:200]
                card = f"""
                <div class="rag-card">
                    <div class="card-header"><span class="match-score">{d[1]:.2f}</span></div>
                    <div class="card-title">{d[0]}</div>
                    <div class="card-content">{txt}...</div>
                </div>
                """
                cards_html += card
            cards_html += "</div>"

            status.markdown(cards_html, unsafe_allow_html=True)
            thought_buffer += f"### 🔍 Retrieval\nRetrieved {len(retrieved_items)} documents.\n\n"

            # Step 3: Reasoning & verification
            status.write("💡 **Step 3: Reasoning & Verification...**")
            context = agent_system.generator.format_context(retrieved_items, doc_collection)
            final_answer = agent_system._generate_with_reasoning(prompt, context)

            passed, feedback = agent_system._self_check_answer(prompt, final_answer, [d[0] for d in retrieved_items], doc_collection)
            if passed:
                status.write("✅ **Self-Check Passed**")
                thought_buffer += f"### ✅ Verification\nPassed.\n"
            else:
                status.write(f"⚠️ **Self-Check Warning:** {feedback}")
                status.write("🔧 **Refining Answer...**")
                final_answer = agent_system._reflect_and_refine(prompt, final_answer, feedback, context)
                thought_buffer += f"### 🔧 Refinement\nTriggered by: {feedback}\n"

            status.update(label="Reasoning Complete", state="complete", expanded=False)

            # Show model's internal thinking as collapsible expander
            if agent_system.generator.last_thinking:
                with st.expander("🧠 Model Thinking", expanded=False):
                    st.markdown(agent_system.generator.last_thinking)
                thought_buffer += f"### 🧠 Model Thinking\n{agent_system.generator.last_thinking}\n\n"

            # Stream final answer
            container = st.empty()
            stream_out = ""
            for char in final_answer:
                stream_out += char
                container.markdown(stream_out + "▌")
                time.sleep(0.005)
            container.markdown(stream_out)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "thoughts": thought_buffer,
                "timestamp": datetime.now().isoformat()
            })
            all_histories[current_id] = st.session_state.messages
            save_history(all_histories)
            st.rerun()

        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"Process Failed: {e}")