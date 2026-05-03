import streamlit as st
import sys, os
import time
import plotly.express as px
import requests
from streamlit_lottie import st_lottie

# Proje dizinini ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from masking.presidio_engine import anonymize_text, build_engines

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

@st.cache_resource
def load_engines():
    return build_engines()

analyzer, anonymizer = load_engines()

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
        start_time = time.time()
        anonymized, results = anonymize_text(analyzer, anonymizer, raw_data)
        duration = time.time() - start_time
        
        with right_panel:
            st.markdown("#### MASKED OUTPUT")
            st.code(anonymized.text, language="text")
            
            if results:
                st.markdown("#### ANALYTICS")
                counts = {}
                for r in results:
                    counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
                
                fig = px.pie(
                    names=list(counts.keys()), 
                    values=list(counts.values()),
                    hole=0.6,
                    color_discrete_sequence=['#64ffda', '#112240']
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color="#8892b0",
                    margin=dict(t=0, b=0, l=0, r=0),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

        # Kutular en altta, geniş yerleşimli
        st.write("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("DETECTIONS", f"{len(results)}")
        # Latency tam ham veri olarak (hiçbir yuvarlama olmadan) ekrana basılıyor
        m2.metric("LATENCY", f"{duration}s")
        m3.metric("INTEGRITY", "VERIFIED")
    else:
        st.error("ERROR: NULL_INPUT")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### CONTROL")
    st.write("USER: OPERATOR")
    st.write("CORE: ACTIVE")
    st.write("---")
    st.caption("ERLIKGATE PROTOCOL 2026")
    # Digital Waste yazısı buradan kaldırıldı.