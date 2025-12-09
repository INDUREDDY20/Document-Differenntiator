import streamlit as st
import io
import json
import nltk
import pandas as pd
import numpy as np
from PIL import Image
import pdfplumber
import docx
from pdf2image import convert_from_bytes
import pytesseract
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import imagehash
from skimage.metrics import structural_similarity as ssim

# ============ SETUP ============
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = get_model()

# ============ ULTIMATE 2025 DESIGN (REAL ICONS + ANIMATIONS) ============
st.set_page_config(page_title="DifferencePro AI • Compare Anything", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800&display=swap');
    
    html, body, .stApp {font-family: 'Space Grotesk', sans-serif; background: #0a0a1f;}
    
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh;}
    
    .header {text-align: center; padding: 3rem 0 2rem;}
    .title {font-size: 5rem; font-weight: 800; background: linear-gradient(90deg, #a8edea, #fed6e3); 
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;}
    .subtitle {color: #e0dfff; font-size: 1.5rem; font-weight: 300; margin-top: 1rem;}
    
    .card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 28px;
        padding: 2.5rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.4);
        transition: all 0.4s ease;
        margin: 1.5rem 0;
    }
    .card:hover {transform: translateY(-12px); box-shadow: 0 30px 70px rgba(102, 126, 234, 0.5);}
    
    .upload-box {
        border: 3px dashed rgba(255,255,255,0.4);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s;
    }
    .upload-box:hover {border-color: #a8edea; background: rgba(168, 237, 234, 0.1);}
    
    .result-metric {
        background: rgba(255,255,255,0.15);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 3.8rem;
        font-weight: 800;
        color: #fff;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #feca57, #48dbfb, #a8edea);
        background-size: 300%;
        color: white;
        border: none;
        border-radius: 50px;
        padding: 18px 50px;
        font-size: 1.3rem;
        font-weight: 700;
        animation: gradient 8s ease infinite;
        box-shadow: 0 15px 35px rgba(255,107,107,0.4);
    }
    @keyframes gradient {0%,100%{background-position:0% 50%} 50%{background-position:100% 50%}}
    
    .sidebar .sidebar-content {background: rgba(10,10,31,0.95); backdrop-filter: blur(15px);}
    .css-1d391kg {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# ============ SIDEBAR WITH ANIMATED ICONS ============
st.sidebar.markdown("""
<div style='text-align:center; padding:2rem 0;'>
<h1 style='color:#a8edea; margin:0;'><i class="fa-solid fa-brain fa-bounce" style="font-size:3rem;"></i></h1>
<h2 style='color:#fff; margin:10px 0;'>DiffPro AI</h2>
<p style='color:#a29bfe; font-size:0.9rem;'>World's Smartest Document Comparator</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Compare Documents", "Features", "About Me"],
    format_func=lambda x: {
        "Compare Documents": "Compare Documents",
        "Features": "Key Features",
        "About Me": "Developer"
    }[x]
)

# ============ HOME: COMPARISON PAGE ============
if page == "Compare Documents":
    st.markdown("<div class='header'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>DiffPro AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload any two files • Get AI-powered insights instantly</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3><i class='fa-solid fa-file-circle-plus' style='color:#a8edea;'></i> Document A</h3>", unsafe_allow_html=True)
        file_a = st.file_uploader("", type=['pdf','docx','txt','xlsx','png','jpg','jpeg'], key="a", label_visibility="collapsed")
        if file_a:
            st.markdown(f"<p style='color:#a8edea;'><i class='fa-solid fa-check-circle'></i> {file_a.name}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3><i class='fa-solid fa-file-circle-plus' style='color:#fed6e3;'></i> Document B</h3>", unsafe_allow_html=True)
        file_b = st.file_uploader("", type=['pdf','docx','txt','xlsx','png','jpg','jpeg'], key="b", label_visibility="collapsed")
        if file_b:
            st.markdown(f"<p style='color:#fed6e3;'><i class='fa-solid fa-check-circle'></i> {file_b.name}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if file_a and file_b:
        if st.button("Run AI Comparison", use_container_width=True):
            with st.spinner("AI is analyzing your documents..."):
                bytes_a = file_a.read()
                bytes_b = file_b.read()
                name_a = file_a.name
                name_b = file_b.name

                # IMAGE COMPARISON
                if name_a.lower().endswith(('png','jpg','jpeg')) and name_b.lower().endswith(('png','jpg','jpeg')):
                    img_a = Image.open(io.BytesIO(bytes_a))
                    img_b = Image.open(io.BytesIO(bytes_b))
                    h1 = imagehash.phash(img_a)
                    h2 = imagehash.phash(img_b)
                    phash_sim = 1 - (h1 - h2) / len(h1.hash)**2
                    g1 = np.array(img_a.convert('L'))
                    g2 = np.array(img_b.convert('L'))
                    ssim_score = ssim(g1, g2, data_range=255)

                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<h2 style='text-align:center; color:#fff;'><i class='fa-solid fa-images'></i> Image Analysis</h2>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"<div class='result-metric'>{phash_sim:.1%}</div><p style='text-align:center; color:#a8edea;'>Visual Match</p>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div class='result-metric'>{ssim_score:.3f}</div><p style='text-align:center; color:#fed6e3;'>Quality Score</p>", unsafe_allow_html=True)
                    with col3:
                        status = "Identical" if phash_sim > 0.95 else "Different"
                        color = "#a8edea" if phash_sim > 0.95 else "#ff6b6b"
                        st.markdown(f"<div class='result-metric' style='color:{color};'>{status}</div><p style='text-align:center; color:#fff;'>Result</p>", unsafe_allow_html=True)
                    st.image([img_a, img_b], width=400)
                    st.markdown("</div>", unsafe_allow_html=True)

                # TEXT COMPARISON
                else:
                    def extract_text(data, filename):
                        name = filename.lower()
                        if name.endswith('.txt'): return data.decode('utf-8')
                        if name.endswith('.docx'):
                            doc = docx.Document(io.BytesIO(data))
                            return "\n".join(p.text for p in doc.paragraphs)
                        if name.endswith('.pdf'):
                            try:
                                with pdfplumber.open(io.BytesIO(data)) as pdf:
                                    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                                if len(text.strip()) > 50: return text
                            except: pass
                            images = convert_from_bytes(data, dpi=200)
                            return "\n".join(pytesseract.image_to_string(img) for img in images)
                        if name.endswith(('.xlsx','.xls')):
                            return pd.read_excel(io.BytesIO(data)).to_string()
                        return ""

                    text_a = extract_text(bytes_a, name_a)
                    text_b = extract_text(bytes_b, name_b)

                    lines_a = text_a.splitlines()
                    lines_b = text_b.splitlines()
                    matcher = SequenceMatcher(None, lines_a, lines_b)
                    ops = matcher.get_opcodes()
                    changes = sum(1 for op in ops if op[0] != 'equal')
                    line_sim = 1 - changes / max(len(lines_a), 1)

                    sents_a = [s for s in nltk.sent_tokenize(text_a) if s.strip()]
                    sents_b = [s for s in nltk.sent_tokenize(text_b) if s.strip()]
                    sem_sim = 0
                    if sents_a and sents_b:
                        emb_a = model.encode(sents_a, convert_to_tensor=True)
                        emb_b = model.encode(sents_b, convert_to_tensor=True)
                        sem_sim = util.cos_sim(emb_a, emb_b).diag().mean().item()

                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<h2 style='text-align:center; color:#fff;'><i class='fa-solid fa-robot'></i> AI Analysis Complete</h2>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"<div class='result-metric' style='color:#a8edea;'>{line_sim:.1%}</div>", unsafe_allow_html=True)
                        st.markdown("<p style='text-align:center; color:#a8edea; font-size:1.3rem;'><i class='fa-solid fa-align-left'></i> Line-by-Line Match</p>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div class='result-metric' style='color:#fed6e3;'>{sem_sim:.1%}</div>", unsafe_allow_html=True)
                        st.markdown("<p style='text-align:center; color:#fed6e3; font-size:1.3rem;'><i class='fa-solid fa-lightbulb'></i> Semantic Meaning Match</p>", unsafe_allow_html=True)

                    st.markdown("<h3 style='color:#fff; text-align:center; margin-top:2rem;'><i class='fa-solid fa-table'></i> Side-by-Side View</h3>", unsafe_allow_html=True)
                    left, right = [], []
                    for tag, i1, i2, j1, j2 in ops[:120]:
                        a_part = lines_a[i1:i2]
                        b_part = lines_b[j1:j2]
                        max_len = max(len(a_part), len(b_part))
                        a_part += [""] * (max_len - len(a_part))
                        b_part += [""] * (max_len - len(b_part))
                        for a, b in zip(a_part, b_part):
                            if tag == 'equal':
                                left.append(a); right.append(b)
                            elif tag == 'delete':
                                left.append(f"<span style='color:#ff6b6b;'>{a}</span>"); right.append("")
                            elif tag == 'insert':
                                left.append(""); right.append(f"<span style='color:#51cf66;'>{b}</span>")
                            elif tag == 'replace':
                                left.append(f"<span style='color:#ff922b;'>{a}</span>"); right.append(f"<span style='color:#ff922b;'>{b}</span>")

                    df = pd.DataFrame({"Document A": left, "Document B": right})
                    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                    st.download_button(
                        "Download Full AI Report",
                        json.dumps({"line_similarity": line_sim, "semantic_similarity": sem_sim, "total_changes": changes}, indent=2),
                        "DiffPro_AI_Report.json",
                        "application/json"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

# ============ FEATURES PAGE ============
elif page == "Features":
    st.markdown("<h1 class='title'>Why DiffPro AI?</h1>", unsafe_allow_html=True)
    features = [
        ("OCR for Scanned PDFs", "fa-solid fa-eye"),
        ("AI Semantic Understanding", "fa-solid fa-brain"),
        ("Excel & Table Comparison", "fa-solid fa-table"),
        ("Image Visual Diff", "fa-solid fa-images"),
        ("Beautiful Modern UI", "fa-solid fa-gem"),
        ("Download Reports", "fa-solid fa-download"),
        ("100% Free & Open Source", "fa-solid fa-heart")
    ]
    cols = st.columns(4)
    for i, (text, icon) in enumerate(features):
        with cols[i % 4]:
            st.markdown(f"<div class='card' style='text-align:center; height:180px;'>", unsafe_allow_html=True)
            st.markdown(f"<h2><i class='{icon}' style='color:#a8edea; font-size:3rem;'></i></h2>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:#fff;'>{text}</h4>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ============ DEVELOPER PAGE ============
elif page == "Developer":
    st.markdown("<h1 class='title'>Indu Reddy</h1>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,2])
    with col1:
        st.image("https://avatars.githubusercontent.com/u/yourid", width=280, caption="Data Scientist")
    with col2:
        st.markdown("""
        <h2 style='color:#a8edea;'>AI Engineer</h2>
        <p style='color:#e0dfff; font-size:1.2rem; line-height:1.8;'>
        Building intelligent, beautiful tools that solve real problems.<br><br>
        This app uses AI, OCR, NLP, and Computer Vision to compare any document format.<br><br>
        <strong>GitHub:</strong> <a href='https://github.com/indureddy20' style='color:#667eea;'>github.com/indureddy20</a><br>
        <strong>Deployed on:</strong> Streamlit Cloud (Free Tier)<br><br>
        This project is 100% open-source and production-ready.
        </p>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("<p style='text-align:center; color:#667eea; margin-top:4rem; font-size:0.9rem;'>© 2025 DifferencePro AI • Made with passion in Bengaluru</p>", unsafe_allow_html=True)
