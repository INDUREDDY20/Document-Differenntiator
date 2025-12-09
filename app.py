import streamlit as st

# MUST BE FIRST — fixes all errors
st.set_page_config(
    page_title="Difference Pro AI - Document Comparator",
    page_icon="magnifying-glass-tilted",
    layout="wide",
    initial_sidebar_state="expanded"
)

import io, json, nltk, pandas as pd, numpy as np
from PIL import Image
import pdfplumber, docx
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

# ============ SOFT & BEAUTIFUL DESIGN ============
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&display=swap');
    html, body, .stApp {font-family: 'Space Grotesk', sans-serif;}
    
    .main {background: linear-gradient(135deg, #e0f7fa 0%, #fff8e1 50%, #f3e5f5 100%); min-height: 100vh;}
    
    .header {text-align: center; padding: 3rem 0 2rem;}
    .title {
        font-size: 5rem; font-weight: 800;
        background: linear-gradient(90deg, #26a69a, #7b1fa2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .subtitle {color: #424242; font-size: 1.5rem; font-weight: 400; margin-top: 1rem;}
    
    .card {
        background: rgba(255,255,255,0.95);
        border-radius: 28px; padding: 2.5rem; margin: 1.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.08);
        transition: all 0.4s ease;
    }
    .card:hover {transform: translateY(-10px); box-shadow: 0 25px 50px rgba(38,166,154,0.2);}
    
    .result-metric {
        background: linear-gradient(45deg, #26a69a, #4db6ac);
        color: white; padding: 2rem; border-radius: 20px;
        text-align: center; font-size: 3.8rem; font-weight: 800;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #26a69a, #7b1fa2);
        color: white; border: none; border-radius: 50px;
        padding: 18px 50px; font-size: 1.3rem; font-weight: 700;
        box-shadow: 0 10px 30px rgba(38,166,154,0.3);
    }
    
    .stFileUploader > div > div > label {color: #26a69a !important; font-weight: 600; font-size: 1.1rem;}
    .stFileUploader > div > div > div {background: #f8fdff !important; border: 2px dashed #26a69a !important; border-radius: 16px;}
    
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #26a69a, #7b1fa2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============ SIDEBAR ============
st.sidebar.markdown("""
<div style='text-align:center; padding:2rem 0;'>
<h2 style='color:white; margin:10px 0;'>DiffPro AI</h2>
<p style='color:#e8f5e8; font-size:0.95rem;'>Smart Document Comparison</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", ["Compare Documents", "Features", "About Me"])

# ============ MAIN COMPARISON PAGE ============
if page == "Compare Documents":
    st.markdown("<div class='header'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>DiffPro AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload any two files • Get instant AI-powered comparison</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>First Document</h3>", unsafe_allow_html=True)
        file_a = st.file_uploader("Choose first file", type=['pdf','docx','txt','xlsx','png','jpg','jpeg'], key="a")
        if file_a: st.success(f"{file_a.name}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Second Document</h3>", unsafe_allow_html=True)
        file_b = st.file_uploader("Choose second file", type=['pdf','docx','txt','xlsx','png','jpg','jpeg'], key="b")
        if file_b: st.success(f"{file_b.name}")
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
                    st.markdown("<h2 style='text-align:center; color:#26a69a;'>Image Comparison Results</h2>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"<div class='result-metric'>{phash_sim:.1%}</div><p style='text-align:center; color:#26a69a;'>Visual Match</p>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div class='result-metric'>{ssim_score:.3f}</div><p style='text-align:center; color:#7b1fa2;'>Quality Score</p>", unsafe_allow_html=True)
                    with col3:
                        status = "Identical" if phash_sim > 0.95 else "Different"
                        color = "#26a69a" if phash_sim > 0.95 else "#ff6b6b"
                        st.markdown(f"<div class='result-metric' style='background:#ff6b6b;color:white;'>{status}</div><p style='text-align:center; color:#424242;'>Result</p>", unsafe_allow_html=True)
                    st.image([img_a, img_b], caption=["Document A", "Document B"], width=400)
                    st.markdown("</div>", unsafe_allow_html=True)

                # TEXT & DOCUMENT COMPARISON
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
                    st.markdown("<h2 style='text-align:center; color:#26a69a;'>AI Analysis Complete</h2>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"<div class='result-metric'>{line_sim:.1%}</div>", unsafe_allow_html=True)
                        st.markdown("<p style='text-align:center; color:#26a69a; font-size:1.3rem;'>Line-by-Line Match</p>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div class='result-metric'>{sem_sim:.1%}</div>", unsafe_allow_html=True)
                        st.markdown("<p style='text-align:center; color:#7b1fa2; font-size:1.3rem;'>Semantic Meaning Match</p>", unsafe_allow_html=True)

                    st.markdown("<h3 style='color:#424242; text-align:center; margin-top:2rem;'>Side-by-Side View</h3>", unsafe_allow_html=True)
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
                        json.dumps({"line_similarity": line_sim, "semantic_similarity": sem_sim, "changes_detected": changes}, indent=2),
                        "DiffPro_AI_Report.json",
                        "application/json"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

# ============ FEATURES PAGE ============
elif page == "Features":
    st.markdown("<div class='header'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Why Choose DiffPro AI?</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    features = [
        ("AI Semantic Comparison", "fa-solid fa-brain", "#26a69a"),
        ("OCR for Scanned PDFs", "fa-solid fa-eye", "#7b1fa2"),
        ("Excel & Table Support", "fa-solid fa-table", "#4db6ac"),
        ("Image Visual Diff", "fa-solid fa-images", "#ff6b6b"),
        ("Beautiful Modern UI", "fa-solid fa-gem", "#feca57"),
        ("Download Reports", "fa-solid fa-download", "#51cf66"),
        ("100% Free Forever", "fa-solid fa-heart", "#ff6b6b")
    ]
    cols = st.columns(4)
    for i, (text, icon, color) in enumerate(features):
        with cols[i % 4]:
            st.markdown(f"<div class='card' style='text-align:center; height:220px;'>", unsafe_allow_html=True)
            st.markdown(f"<h2><i class='{icon}' style='color:{color}; font-size:3.5rem;'></i></h2>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:#424242;'>{text}</h4>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ============ ABOUT ME PAGE ============
elif page == "About Me":
    st.markdown("<div class='header'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Indu Reddy</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,2])
    with col1:
        st.image("https://avatars.githubusercontent.com/u/123456789?v=4", width=280, caption="Data Analyst Intern")
    with col2:
        st.markdown("""
        <h2 style='color:#26a69a;'> Aspiring AI Engineer • Bengaluru</h2>
        <p style='color:#424242; font-size:1.2rem; line-height:1.8;'>
        Passionate about building intelligent, beautiful tools that solve real problems.<br><br>
        This app uses AI, OCR, NLP, and Computer Vision to compare any document format.<br><br>
        <strong>GitHub:</strong> <a href='https://github.com/indureddy20' style='color:#26a69a;'>github.com/indureddy20</a><br>
        <strong>Deployed on:</strong> Streamlit Cloud • 100% Free & Open Source
        </p>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("<p style='text-align:center; color:#e8f5e8; margin-top:4rem; font-size:0.9rem;'>© 2025 • Made with love and passion</p>", unsafe_allow_html=True)
