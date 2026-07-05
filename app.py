import streamlit as st
import requests
import re
import html

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE = "http://127.0.0.1:8000"
ADMIN_USERNAME = "admin"

# ── Markdown → HTML helper ────────────────────────────────────────────────────

def render_markdown(text: str) -> str:
    """Clean LLM output and convert basic markdown to safe HTML."""
    # Strip any complete leaked source metadata
    text = re.sub(r'__SOURCES__.*?__END_SOURCES__', '', text, flags=re.DOTALL)
    # Strip any dangling prefix/suffix keywords safely
    text = text.replace("__SOURCES__", "").replace("__END_SOURCES__", "")
    # Strip "Source: ..." lines added by the LLM
    text = re.sub(r'\s*Source:.*?(\n|$)', '', text)
    # Strip connector noise at start
    text = re.sub(r'^connector to the original connection\.?\s*', '', text)
    # HTML-escape FIRST so characters like < > & don't break the bubble HTML
    text = html.escape(text)

    # Now apply markdown → HTML (these won't get double-escaped)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*(.+?)\*(?!\*)', r'<em>\1</em>', text)
    text = text.replace('\n', '<br>')
    return text.strip()

# ── Streaming token generator ─────────────────────────────────────────────────

def token_generator(question: str, sources_ref: list):
    try:
        with requests.post(
            f"{API_BASE}/chat/stream",
            json={"question": question},
            stream=True,
            timeout=120
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if not decoded.startswith("data: "):
                    continue
                data = decoded[6:]
                if data == "[DONE]":
                    return
                if data.startswith("__SOURCES__") and "__END_SOURCES__" in data:
                    raw = data.replace("__SOURCES__", "").replace("__END_SOURCES__", "")
                    sources_ref.extend([s.strip() for s in raw.split("|||") if s.strip()])
                    continue
                yield data
    except Exception as e:
        yield f"\n\n⚠️ Connection error: {e}"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HVAC Support Portal",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Theme ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #f8fafc !important; }
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }

.navbar { display:flex; align-items:center; justify-content:space-between; padding:14px 0; border-bottom:1px solid #e2e8f0; margin-bottom:28px; }
.nav-logo { font-size:1.25rem; font-weight:800; color:#1e293b; letter-spacing:-0.5px; }
.nav-logo span { color:#2563eb; }
.nav-badge { background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8; padding:4px 14px; border-radius:20px; font-size:0.78rem; font-weight:600; }

.hero-title { font-size:2.6rem; font-weight:800; color:#0f172a; line-height:1.15; letter-spacing:-1px; }
.hero-title span { color:#2563eb; }
.hero-sub { color:#64748b; font-size:1rem; margin-top:8px; margin-bottom:32px; }

.login-card { background:#ffffff; border:1px solid #e2e8f0; border-radius:20px; padding:44px 40px; box-shadow:0 4px 24px rgba(0,0,0,0.06); }

.stat-card { background:#ffffff; border:1px solid #e2e8f0; border-radius:16px; padding:24px 20px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.04); }
.stat-number { font-size:2.4rem; font-weight:800; color:#2563eb; line-height:1; }
.stat-label { color:#64748b; font-size:0.8rem; font-weight:600; margin-top:6px; text-transform:uppercase; letter-spacing:0.06em; }

.section-header { font-size:1.3rem; font-weight:700; color:#0f172a; margin-bottom:4px; }
.section-sub { color:#64748b; font-size:0.88rem; margin-bottom:20px; }

.chat-user { display:flex; justify-content:flex-end; margin:10px 0; }
.chat-user-bubble { background:#2563eb; color:#ffffff; padding:11px 17px; border-radius:18px 18px 4px 18px; max-width:72%; font-size:0.93rem; line-height:1.55; box-shadow:0 2px 10px rgba(37,99,235,0.2); }
.chat-bot { display:flex; justify-content:flex-start; margin:10px 0; align-items:flex-start; gap:10px; }
.bot-avatar { width:32px; height:32px; min-width:32px; background:#eff6ff; border:1px solid #bfdbfe; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:14px; }
.chat-bot-bubble { background:#ffffff; border:1px solid #e2e8f0; color:#1e293b; padding:12px 17px; border-radius:4px 18px 18px 18px; max-width:72%; font-size:0.93rem; line-height:1.65; box-shadow:0 2px 8px rgba(0,0,0,0.04); }

.empty-state { text-align:center; padding:70px 20px; }
.empty-state .icon { font-size:3rem; margin-bottom:14px; }
.empty-state .title { font-size:1.1rem; font-weight:600; color:#475569; }
.empty-state .hint { font-size:0.88rem; margin-top:6px; color:#94a3b8; }

.info-banner { background:#eff6ff; border:1px solid #bfdbfe; border-radius:12px; padding:14px 18px; margin-bottom:20px; color:#1e40af; font-size:0.88rem; line-height:1.5; }
.file-info { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:12px 16px; margin-top:8px; }
.file-name { color:#1e293b; font-weight:600; font-size:0.9rem; }
.file-meta { color:#94a3b8; font-size:0.82rem; }

.stButton > button { background:#2563eb !important; color:#ffffff !important; border:none !important; border-radius:10px !important; padding:10px 22px !important; font-weight:600 !important; font-size:0.93rem !important; transition:all 0.2s ease !important; box-shadow:0 2px 8px rgba(37,99,235,0.25) !important; }
.stButton > button:hover { background:#1d4ed8 !important; box-shadow:0 4px 16px rgba(37,99,235,0.35) !important; transform:translateY(-1px) !important; }

.stTextInput > label { color:#374151 !important; font-weight:500 !important; font-size:0.88rem !important; }
.stTextInput > div > div > input { background:#ffffff !important; border:1.5px solid #d1d5db !important; border-radius:10px !important; color:#111827 !important; font-size:0.93rem !important; padding:10px 14px !important; }
.stTextInput > div > div > input:focus { border-color:#2563eb !important; box-shadow:0 0 0 3px rgba(37,99,235,0.12) !important; }
.stTextInput > div > div > input::placeholder { color:#9ca3af !important; }

.stSelectbox > label { color:#374151 !important; font-weight:500 !important; font-size:0.88rem !important; }
.stSelectbox > div > div { background:#ffffff !important; border:1.5px solid #d1d5db !important; border-radius:10px !important; color:#111827 !important; }

div[data-testid="stFileUploader"] { background:#f8fafc !important; border:2px dashed #cbd5e1 !important; border-radius:12px !important; }
.stExpander { background:#ffffff !important; border:1px solid #e2e8f0 !important; border-radius:10px !important; }
div[data-testid="stMetric"] { background:#ffffff !important; border:1px solid #e2e8f0 !important; border-radius:12px !important; padding:16px !important; }
div[data-testid="stMetricValue"] { color:#2563eb !important; font-weight:700 !important; }
div[data-testid="stAlert"] { border-radius:10px !important; }
.stSpinner > div { border-top-color:#2563eb !important; }
hr { border-color:#e2e8f0 !important; margin:24px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── API Helpers ───────────────────────────────────────────────────────────────

def fetch_stats():
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"total_chunks": "—", "total_documents": "—", "documents": []}


def upload_file_to_api(file_bytes, filename, content_type):
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (filename, file_bytes, content_type)},
            timeout=300
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── PAGE: Login ───────────────────────────────────────────────────────────────

def page_login():
    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        st.markdown("""
        <div style='text-align:center; margin-bottom:15px; margin-top:5px;'>
            <div class='hero-title' style='font-size:2.2rem;'>HVAC <span>Support</span> Portal</div>
            <div class='hero-sub' style='font-size:0.9rem; margin-bottom:15px;'>Industrial Heating & Cooling Intelligence Platform</div>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("<p style='color:#374151; font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;'>Select Role</p>", unsafe_allow_html=True)
            role = st.selectbox("Select your role", ["👤 Customer (Technician / Facility Manager)", "🔐 Administrator"], label_visibility="collapsed", key="role_select")

            is_admin = "Administrator" in role

            if is_admin:
                st.markdown("<hr style='margin: 15px 0 !important;'>", unsafe_allow_html=True)
                st.markdown("<p style='color:#374151; font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;'>Admin Credentials</p>", unsafe_allow_html=True)
                username = st.text_input("Username", placeholder="admin", key="login_user")
                password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔐  Login to Dashboard", use_container_width=True, key="admin_login_btn"):
                    if username == ADMIN_USERNAME and len(password) > 0:
                        st.session_state.logged_in = True
                        st.session_state.role = "admin"
                        st.rerun()
                    else:
                        st.error("Username must be **admin** and password cannot be empty.")
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("💬  Enter Chat Assistant", use_container_width=True, key="customer_enter_btn"):
                    st.session_state.logged_in = True
                    st.session_state.role = "customer"
                    st.rerun()



# ── PAGE: Admin Dashboard ─────────────────────────────────────────────────────

def page_admin():
    
    st.markdown("<div class='section-header'>Knowledge Base Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Real-time statistics from your ChromaDB vector store</div>", unsafe_allow_html=True)

    stats = fetch_stats()
    doc_list = stats.get("documents", [])
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='stat-card'><div class='stat-number'>{stats.get('total_documents', 0)}</div><div class='stat-label'>📄 Documents Indexed</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='stat-card'><div class='stat-number'>{stats.get('total_chunks', 0)}</div><div class='stat-label'>🧩 Knowledge Chunks</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='stat-card'><div class='stat-number'>{len(doc_list)}</div><div class='stat-label'>📚 Unique Manuals</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if doc_list:
        with st.expander(f"📂 Indexed Documents ({len(doc_list)})"):
            for i, doc in enumerate(doc_list, 1):
                st.markdown(f"**{i}.** {doc}")

    st.divider()
    st.markdown("<div class='section-header'>Upload Files</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Upload equipment manuals (PDF) or reference images. PDFs are chunked, embedded, and stored for the chatbot.</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop files here or click to browse", type=["pdf", "png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_col, btn_col = st.columns([3, 1])
        with file_col:
            st.markdown(f"""
            <div class='file-info'>
                <div class='file-name'>📄 {uploaded_file.name}</div>
                <div class='file-meta'>{uploaded_file.type} &nbsp;•&nbsp; {uploaded_file.size / 1024:.1f} KB</div>
            </div>
            """, unsafe_allow_html=True)
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚡ Process & Ingest", use_container_width=True, key="ingest_btn"):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    result = upload_file_to_api(uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type)
                if "error" in result:
                    st.error(f"Upload failed: {result['error']}")
                else:
                    st.success(f"✅ **{result.get('filename', '')}** ingested successfully!")
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Chunks Created", result.get("chunk_count", 0))
                    with m2:
                        st.metric("Embeddings Generated", result.get("embeddings_created", 0))
                    st.rerun()


# ── PAGE: Customer Chatbot ─────────────────────────────────────────────────────

def page_customer():
    # ── Navbar ──
    nav_col, clear_col, exit_col = st.columns([0.65, 0.22, 0.13])
    with nav_col:
        st.markdown("""
        <div class='navbar'>
            <div class='nav-logo'><span></span> Support Portal</div>
        </div>
        """, unsafe_allow_html=True)



    # ── Render past history ──
    if not st.session_state.chat_history and not st.session_state.pending_question:
        st.markdown("""
        <div class='empty-state'>
            <div class='icon'>🤖</div>
            <div class='title'>How can I help you today?</div>
            <div class='hint'>Try: "How do I resolve Error Code 41?" or "What are the filter replacement intervals?"</div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class='chat-user'>
                <div class='chat-user-bubble'>{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            rendered = render_markdown(msg["content"])
            st.markdown(f"""
            <div class='chat-bot'>
                <div class='bot-avatar'>🤖</div>
                <div class='chat-bot-bubble'>{rendered}</div>
            </div>
            """, unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📚 View Source Chunks"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.caption(src[:500] + ("..." if len(src) > 500 else ""))
                        if i < len(msg["sources"]):
                            st.divider()

    # ── KEY FIX: Streaming happens HERE — between history and form ──
    if st.session_state.pending_question:
        question = st.session_state.pending_question

        # User bubble for the pending question
        st.markdown(f"""
        <div class='chat-user'>
            <div class='chat-user-bubble'>{question}</div>
        </div>
        """, unsafe_allow_html=True)

        sources_found = []
        full_answer_raw = ""

        # Stream with direct st.empty() control — guaranteed token collection
        bot_col1, bot_col2 = st.columns([0.04, 0.96])
        with bot_col1:
            st.markdown("<div class='bot-avatar' style='margin-top:6px;'>🤖</div>", unsafe_allow_html=True)
        with bot_col2:
            answer_box = st.empty()
            try:
                with requests.post(
                    f"{API_BASE}/chat/stream",
                    json={"question": question},
                    stream=True,
                    timeout=120
                ) as resp:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        decoded = line.decode("utf-8")
                        if not decoded.startswith("data: "):
                            continue
                        data = decoded[6:]
                        if data == "[DONE]":
                            break
                        if data.startswith("__SOURCES__") and "__END_SOURCES__" in data:
                            raw = data.replace("__SOURCES__", "").replace("__END_SOURCES__", "")
                            sources_found.extend([s.strip() for s in raw.split("|||") if s.strip()])
                            continue
                        full_answer_raw += data
                        # Live update — cursor blink effect
                        answer_box.markdown(full_answer_raw + "▌")
                # Final render without cursor
                answer_box.markdown(full_answer_raw)
            except Exception as e:
                full_answer_raw = f"⚠️ Connection error: {e}"
                answer_box.markdown(full_answer_raw)

        # Save raw text to history (render_markdown called only at display time)
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({
            "role": "bot",
            "content": full_answer_raw,
            "sources": sources_found
        })
        st.session_state.pending_question = None
        st.rerun()
        return



    # ── Input form — only shown when NOT streaming ──
    if not st.session_state.pending_question:
        user_input = st.chat_input("Ask a question...")
        if user_input:
            st.session_state.pending_question = user_input.strip()
            st.rerun()



# ── Router ─────────────────────────────────────────────────────────────────────

def main():
    if not st.session_state.logged_in:
        page_login()
    elif st.session_state.role == "admin":
        page_admin()
    elif st.session_state.role == "customer":
        page_customer()

if __name__ == "__main__":
    main()
