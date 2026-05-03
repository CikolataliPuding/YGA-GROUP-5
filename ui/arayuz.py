import streamlit as st
import sys, os
import time
import plotly.express as px
import requests
from streamlit_lottie import st_lottie

# Proje dizinini ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from masking.fast_masker import mask_text

# 1. Sayfa Yapılandırması
st.set_page_config(page_title="ERLIKGATE INTEL", layout="wide")

# 2. Lottie Yükleyici
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_shield = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_6aYlHk.json")

# 3. TEKNİK CYBER-UI TASARIMI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(180deg, #050a10 0%, #0a192f 100%);
        color: #ccd6f6;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    }

    [data-testid="stSidebar"] {
        background-color: #020c1b;
        border-right: 1px solid #112240;
    }

    .stMetric, .metric-container, div.stCode {
        background: rgba(17, 34, 64, 0.7) !important;
        border: 1px solid #233554 !important;
        border-left: 4px solid #64ffda !important;
        border-radius: 2px !important;
        padding: 20px !important;
    }

    h1 {
        color: #64ffda !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }

    .stButton>button {
        background: transparent !important;
        color: #64ffda !important;
        border: 1px solid #64ffda !important;
        border-radius: 0px !important;
        width: 100%;
    }

    .stButton>button:hover {
        background: rgba(100, 255, 218, 0.1) !important;
    }

    .stTextArea textarea {
        background-color: #020c1b !important;
        border: 1px solid #233554 !important;
        color: #64ffda !important;
        border-radius: 0px !important;
        font-family: 'Consolas', monospace !important;
    }
    </style>
    """, unsafe_allow_html=True)

_DECISION_COLORS = {
    "TEHDIT":   "#ff4444",
    "KAYTARMA": "#ffaa00",
    "GUVENLI":  "#64ffda",
}

# --- HEADER ---
header_col1, header_col2 = st.columns([0.8, 0.2])
with header_col1:
    st.title("ERLIKGATE SYSTEM")
    st.write("STATUS: ACTIVE // PROTOCOL: KVKK_V2")
with header_col2:
    if lottie_shield:
        st_lottie(lottie_shield, height=80, key="shield")

st.write("---")

# --- ANA PANEL ---
left_panel, right_panel = st.columns([0.5, 0.5], gap="large")

with left_panel:
    st.markdown("#### DATA INGESTION")
    raw_data = st.text_area("RAW_STREAM:", height=350, placeholder="Enjekte edilecek veri bekleniyor...")
    execute = st.button("EXECUTE")

if execute:
    if raw_data:
        # — PII maskeleme —
        t_mask = time.perf_counter()
        masked_text, entities = mask_text(raw_data)
        masking_ms = (time.perf_counter() - t_mask) * 1000

        # — Sınıflandırma —
        gateway_online = True
        classify_result: dict = {}
        try:
            resp = requests.post(
                "http://127.0.0.1:8000/classify",
                json={"text": raw_data},
                timeout=5,
            )
            classify_result = resp.json()
        except Exception:
            gateway_online = False

        e2e_ms = classify_result.get("e2e_ms", 0.0)

        with right_panel:
            # — Üst: MASKED OUTPUT —
            st.markdown("#### MASKED OUTPUT")
            st.code(masked_text, language="text")

            if entities:
                st.markdown("#### ANALYTICS")
                counts: dict[str, int] = {}
                for ent in entities:
                    counts[ent["type"]] = counts.get(ent["type"], 0) + 1
                fig = px.pie(
                    names=list(counts.keys()),
                    values=list(counts.values()),
                    hole=0.6,
                    color_discrete_sequence=["#64ffda", "#112240"],
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#8892b0",
                    margin=dict(t=0, b=0, l=0, r=0),
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.write("---")

            # — Alt: THREAT ANALYSIS —
            st.markdown("#### THREAT ANALYSIS")

            if not gateway_online:
                st.markdown(
                    '<div style="background:#ff444422;border:2px solid #ff4444;'
                    'border-radius:4px;padding:12px;text-align:center;">'
                    '<span style="color:#ff4444;font-size:20px;font-weight:700;'
                    'letter-spacing:3px;">GATEWAY: OFFLINE</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                decision      = classify_result.get("decision", "N/A")
                confidence    = classify_result.get("confidence", 0.0)
                inference_ms  = classify_result.get("inference_ms", 0.0)
                honeypot_sess = classify_result.get("honeypot_session")

                color = _DECISION_COLORS.get(decision, "#8892b0")
                st.markdown(
                    f'<div style="background:{color}22;border:2px solid {color};'
                    f'border-radius:4px;padding:14px;text-align:center;margin-bottom:12px;">'
                    f'<span style="color:{color};font-size:26px;font-weight:700;'
                    f'letter-spacing:4px;">{decision}</span></div>',
                    unsafe_allow_html=True,
                )

                ta1, ta2, ta3 = st.columns(3)
                ta1.metric("CONFIDENCE",  f"{confidence * 100:.1f}%")
                ta2.metric("INFERENCE",   f"{inference_ms:.2f}ms")
                ta3.metric("E2E",         f"{e2e_ms:.2f}ms")

                if honeypot_sess:
                    st.info(f"HONEYPOT SESSION: {honeypot_sess}")

        # — Alt metrik bant —
        st.write("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("DETECTIONS",  str(len(entities)))
        m2.metric("MASKING_MS",  f"{masking_ms:.2f}ms")
        m3.metric("CLASSIFY_MS", f"{e2e_ms:.2f}ms" if gateway_online else "OFFLINE")
    else:
        st.error("ERROR: NULL_INPUT")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### CONTROL")
    st.write("USER: OPERATOR")
    st.write("CORE: ACTIVE")
    st.write("---")
    st.caption("ERLIKGATE PROTOCOL 2026")
