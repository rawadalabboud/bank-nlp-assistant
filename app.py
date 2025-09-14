# app.py
import os
import json
import requests
import streamlit as st

# Inject custom CSS
st.markdown("""
    <style>
    /* Global text */
    body, .stMarkdown, .stTextInput, .stSelectbox, .stSlider label {
        color: #111 !important;  /* dark grey */
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #1a1a1a !important; /* dark background */
        color: white !important; /* white text */
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #444 !important;
        color: #fff !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #111 !important;
        font-weight: 600;
    }

    /* Answer box */
    .stAlert, .stText {
        color: #222 !important;
        font-size: 16px !important;
    }
    
    /* Prediction highlight */
    .pred-box {
        background-color: #f0f4ff;   /* soft light blue */
        color: #1a1a1a !important;  /* dark text */
        font-size: 16px;
        font-weight: 600;
        padding: 8px 12px;
        border-radius: 6px;
        display: inline-block;
        margin-top: 8px;
    }
    .pred-box span {
        color: #0052cc;  /* nice blue for emphasis */
        font-weight: bold;
    }

    </style>
""", unsafe_allow_html=True)




API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

st.set_page_config(page_title="Bank NLP Assistant", page_icon="💳", layout="centered")

# ----------------- minimal styling -----------------
st.markdown("""
<style>
    .stApp {background: linear-gradient(180deg, #f8fafc 0%, #ffffff 60%);}
    .title {font-size: 2rem; font-weight: 800; letter-spacing: .2px;}
    .subtle {color: #64748b; font-size: 0.9rem;}
    .pill {display:inline-block; padding:6px 10px; border-radius:999px; background:#f1f5f9; margin-right:6px;}
    .source {font-size:.9rem; color:#475569;}
    .codebox {background:#0b1021; color:#e2e8f0; padding:10px 12px; border-radius:10px; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">💳 Bank NLP Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">DistilBERT intent classifier + RAG FAQ with cross-encoder reranking</div>', unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3 = st.tabs(["🧠 Intent", "📚 FAQ Answer (RAG)", "🩺 Health"])

# ----------------- Helpers -----------------
def pretty(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False)

def post_json(path, payload):
    url = f"{API_URL}{path}"
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def get_json(path):
    url = f"{API_URL}{path}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

# ----------------- Intent tab -----------------
with tab1:
    st.subheader("Intent Classification")
    txt = st.text_area("Enter a user query", value="I lost my bank card yesterday, what should I do?", height=90)

    colA, colB = st.columns([1,1])
    with colA:
        k = st.slider("Top-K intents", 1, 10, 5)
    with colB:
        run_btn = st.button("Classify")

    if run_btn and txt.strip():
        resp = post_json("/classify", {"text": txt, "k": k})

        # Try both shapes: {"best": {"label","score"}, ...} OR just {"top_k":[...]}
        top_k = resp.get("top_k", [])
        best = resp.get("best") or (top_k[0] if top_k else {})

        # Support keys "label"/"intent" and "score"/"confidence"
        pred = best.get("label") or best.get("intent") or "—"
        conf = float(best.get("score") or best.get("confidence") or 0.0)

        st.markdown(
            f'<div class="pred-box">Predicted: <span>{pred}</span> • confidence: <span>{conf:.2f}</span></div>',
            unsafe_allow_html=True
        )

        if top_k:
            st.write("")
            st.caption("Top-K details")
            # small, clean table
            st.dataframe(
                [{"rank": i+1,
                  "intent": x.get("label") or x.get("intent"),
                  "score": round(float(x.get("score") or x.get("confidence") or 0.0), 4)}
                 for i, x in enumerate(top_k)],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No predictions returned.")

# ----------------- RAG tab -----------------
with tab2:
    st.subheader("FAQ Assistant (RAG + rerank)")
    q = st.text_area("Ask a question", value="ATM swallowed my card, what should I do?", height=90)
    col1, col2 = st.columns([1,1])
    with col1:
        k_retr = st.slider("Retrieve (k)", 10, 60, 20, step=5)
    with col2:
        k_final = st.slider("Return top (k)", 1, 10, 5)

    colx, coly = st.columns([1,1])
    with colx:
        answer_btn = st.button("Get Answer")
    with coly:
        debug_btn = st.button("Debug Scores")

    if answer_btn and q.strip():
        payload = {"text": q, "k_retr": k_retr, "k_final": k_final}
        resp = post_json("/answer", payload)
        # temporarily override via query params environment (optional)
        os.environ["RERANK_K_RETR"] = str(k_retr)
        os.environ["RERANK_K_FINAL"] = str(k_final)
        resp = post_json("/answer", payload)
        st.write(resp.get("answer", ""))
        srcs = resp.get("sources", [])
        if srcs:
            st.markdown("**Sources**")
            for s in srcs:
                st.markdown(f"- <span class='source'>**{s.get('heading','')}** — _{s.get('file','')}_</span><br><span class='subtle'>{s.get('source','')}</span>", unsafe_allow_html=True)
        with st.expander("Raw response"):
            st.json(resp)

    if debug_btn and q.strip():
        dbg = post_json("/answer_debug", {"text": q, "k_retr": k_retr, "k_final": k_final})
        st.markdown("**Top rerank scores**")
        st.code(pretty(dbg.get("top_scores", [])), language="json")
        st.markdown("**Top headings**")
        st.code(pretty(dbg.get("top_headings", [])), language="json")
        with st.expander("Raw debug payload"):
            st.json(dbg)

# ----------------- Health tab -----------------
with tab3:
    if st.button("Check API"):
        try:
            h = get_json("/health")
            st.success("API reachable ✅")
            st.code(pretty(h), language="json")
        except Exception as e:
            st.error(f"API not reachable: {e}")
    st.caption("Tip: run your API with `uvicorn api.main:app --reload --port 8001` and set `API_URL` if needed.")
