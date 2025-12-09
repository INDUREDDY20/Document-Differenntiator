import streamlit as st

# ============ THIS MUST BE THE VERY FIRST STREAMLIT COMMAND ============
st.set_page_config(
    page_title="DiffPro AI - Document Comparator",
    page_icon="magnifying-glass-tilted",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ NOW IMPORT EVERYTHING ELSE ============
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

# ============ GORGEOUS DESIGN ============
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&display=swap');
    html, body, .stApp {font-family: 'Space Grotesk', sans-serif;}
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh;}
    
    .header {text-align: center; padding: 3rem 0 2rem;}
    .title {font-size: 5rem; font-weight: 800; background: linear-gradient(90deg, #a8edea, #fed6e3);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;}
    .subtitle {color: #e0dfff; font-size: 1.5rem; font-weight: 300; margin-top: 1rem;}
    
    .card {
        background: rgba(255,255,255,0.1);
        border-radius: 28px; padding: 2.5rem; margin: 1.5rem 0;
        border: 1px solid rgba(255,255,255,0.18);
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.4);
        transition: all 0.4s ease;
    }
    .card:hover {transform: translateY(-12px); box-shadow: 0 30px 70px rgba(102,126,234,0.5);}
    
    .result-metric {
        background: rgba(255,255,255,0.15); padding: 2rem; border-radius: 20px;
        text-align: center; font-size: 3.8rem; font-weight: 800; color: #fff;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #feca57, #48dbfb, #a8edea);
        background-size: 300%; color: white; border: none; border-radius: 50px;
        padding: 18px 50px; font-size: 1.3rem; font-weight: 700;
        animation: gradient 8s ease infinite;
        box-shadow: 0 15px 35px rgba(255,107,107,0.4);
    }
    @keyframes gradient {0%,100%{background-position:0% 50%} 50%{background-position:100% 50%}}
    
    .sidebar .sidebar-content {background: rgba(10,10,31,0.95); backdrop-filter: blur(15px);}
</style>
""", unsafe_allow_html=True)

# ============ SIDEBAR ============
st.sidebar.markdown("""
<div style='text-align:center; padding:2rem 0;'>
<h1 style='color:#a8edea; margin:0;'></h1>
<h2 style='color:#fff; margin:10px 0;'>DiffPro AI</h2>
<p style='color:#a29bfe; font-size:0.9rem;'>Smartest Document Comparator</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", ["Compare Documents", "Features", "About Me"])

# ============ MAIN PAGE ============
if page == "Compare Documents":
    st.markdown("<div class='header'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>DiffPro AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload any two files • Get AI insights instantly</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>First Document</h3>", unsafe_allow_html=True)
        file_a = st.file_uploader("", type=['pdf','docx','txt','xlsx','png','jpg','jpeg'], key="a", label_visibility="collapsed")
        if file_a: st.success(f"{file_a.name}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Second Document</h3>", unsafe_allow_html=True)
        file_b = st.file_uploader("", type=['pdf','docx','txt','xlsx','png','jpg','jpeg'], key="b", label_visibility="collapsed")
        if file_b: st.success(f"{file_b.name}")
        st.markdown("</div>", unsafe_allow_html=True)

    if file_a and file_b:
        if st.button("Run AI Comparison", use_container_width=True):
            with st.spinner("Analyzing..."):
                bytes_a = file_a.read()
                bytes_b = file_b.read()
                name_a = file_a.name
                name_b = file_b.name

                if name_a.lower().endswith(('png','jpg','jpeg')) and name_b.lower().endswith(('png','jpg','jpeg')):
                    # Image comparison logic
                    img_a = Image.open(io.BytesIO(bytes_a))
                    img_b = Image.open(io.BytesIO(bytes_b))
                    h1 = imagehash.phash(img_a)
                    h2 = imagehash.phash(img_b)
                    phash_sim = 1 - (h1 - h2) / len(h1.hash)**2
                    g1 = np.array(img_a.convert('L'))
                    g2 = np.array(img_b.convert('L'))
                    ssim_score = ssim(g1, g2, data_range=255)

                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"<div class='result-metric'>{phash_sim:.1%}</div><p style='text-align:center; color:#a8edea;'>Visual Match</p>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div class='result-metric'>{ssim_score:.3f}</div><p style='text-align:center; color:#fed6e3;'>Quality Score</p>", unsafe_allow_html=True)
                    st.image([img_a, img_b], width=400)
                    st.markdown("</div>", unsafe_allow_html=True)

                else:
                    # Full text comparison logic
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
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"<div class='result-metric' style='color:#a8edea;'>{line_sim:.1%}</div>", unsafe_allow_html=True)
                        st.markdown("<p style='text-align:center; color:#a8edea; font-size:1.3rem;'>Line-by-Line</p>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div class='result-metric' style='color:#fed6e3;'>{sem_sim:.1%}</div>", unsafe_allow_html=True)
                        st.markdown("<p style='text-align:center; color:#fed6e3; font-size:1.3rem;'>Semantic Match</p>", unsafe_allow_html=True)

                    st.markdown("<h3 style='color:#fff; text-align:center;'>Side-by-Side</h3>", unsafe_allow_html=True)
                    left, right = [], []
                    for tag, i1, i2, j1, j2 in ops[:100]:
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
                    
                    st.download_button("Download Report", json.dumps({"line": line_sim, "semantic": sem_sim}, indent=2), "report.json")
                    st.markdown("</div>", unsafe_allow_html=True)

# ============ OTHER PAGES ============
elif page == "Features":
    st.markdown("<h1 class='title'>Features</h1>", unsafe_allow_html=True)
    st.markdown("All features working perfectly!")

elif page == "About Me":
    st.markdown("<h1 class='title'>Indu Reddy</h1>", unsafe_allow_html=True)
    st.markdown("Your amazing developer profile here!")

st.sidebar.markdown("<p style='text-align:center; color:#667eea; margin-top:4rem;'>© 2025 DiffPro AI</p>", unsafe_allow_html=True)
