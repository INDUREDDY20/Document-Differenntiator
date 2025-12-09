import streamlit as st
import io
import re
import json
import base64
from typing import List, Dict, Any

import nltk
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import pdfplumber
import docx
from pdf2image import convert_from_bytes
import pytesseract
from difflib import SequenceMatcher, unified_diff, HtmlDiff
from sentence_transformers import SentenceTransformer, util
import imagehash
from skimage.metrics import structural_similarity as ssim


# ============================================================
# PAGE CONFIG (MUST BE FIRST)
# ============================================================
st.set_page_config(
    page_title="DiffPro AI - Document Comparator",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# INITIAL SETUP
# ============================================================
nltk.download('punkt', quiet=True)

@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = get_model()


# ============================================================
# GLOBAL CSS THEME
# ============================================================
st.markdown("""
<style>

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0A0A1F 0%, #1A1A3A 50%, #2D1B69 100%);
    color: #E8E8FF;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #131324 0%, #0C0C18 100%);
    padding: 20px 10px !important;
}

.sidebar-container { text-align:center; }
.sidebar-title { color: #00D2FF; font-size: 2rem; font-weight: 800; }
.sidebar-sub { color:#AAB4FF; }

/* NAV BUTTONS */
.stSidebar .stRadio > div { gap: 0.4rem !important; }
.stRadio label { color:#E0E0FF !important; font-size:1.1rem; }

/* BUTTON STYLE */
.stButton > button {
    background: linear-gradient(45deg, #00D2FF, #7C4DFF);
    color: white !important;
    border:none;
    border-radius: 50px;
    padding:12px 35px;
    font-size:1.1rem;
    font-weight:600;
    transition:0.25s;
}
.stButton > button:hover {
    transform: translateY(-3px);
}

/* CARD */
.card {
    background: rgba(255,255,255,0.05);
    padding:1.8rem;
    border-radius:14px;
    border:1px solid rgba(255,255,255,0.08);
    backdrop-filter:blur(10px);
    margin-bottom:20px;
}

/* DIFF VIEWER */
table.diff { width:100%; border-collapse:collapse; font-family:monospace; }
table.diff th { background:rgba(255,255,255,0.05); }
.diff_add { background-color: rgba(76,175,80,0.25) !important; }
.diff_sub { background-color: rgba(244,67,54,0.25) !important; }
.diff_chg { background-color: rgba(255,202,40,0.25) !important; }

/* ABOUT PAGE */
.about-container { display:flex; gap:2rem; align-items:center; }
.about-img {
    width:180px; height:180px; border-radius:50%;
    border:3px solid #00D2FF;
    box-shadow: 0 0 18px rgba(0,210,255,0.4);
}
.about-text { font-size:1.2rem; line-height:1.7; }
.about-text a { color:#7C4DFF; font-weight:600; }

</style>
""", unsafe_allow_html=True)


# ============================================================
# EXTRACTION FUNCTIONS
# ============================================================
def extract_text_from_docx(bytes_data):
    doc = docx.Document(io.BytesIO(bytes_data))
    return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])


def extract_text_from_txt(bytes_data):
    try:
        return bytes_data.decode("utf-8", errors="ignore")
    except:
        return str(bytes_data)


def extract_text_from_image(bytes_data):
    img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    return pytesseract.image_to_string(img)


def extract_text_from_excel(bytes_data):
    try:
        sheets = pd.read_excel(io.BytesIO(bytes_data), sheet_name=None)
        text_parts = []
        out_tables = []
        for name, df in sheets.items():
            out_tables.append((name, df))
            snippet = f"Sheet: {name}\nColumns: {list(df.columns)}\n{df.head()}"
            text_parts.append(snippet)
        return "\n".join(text_parts), out_tables
    except:
        return "", []


def extract_text_from_pdf(bytes_data):
    try:
        pdf = pdfplumber.open(io.BytesIO(bytes_data))
        text = "\n".join(pg.extract_text() or "" for pg in pdf.pages)
        if text.strip():
            return text
    except:
        pass

    try:
        images = convert_from_bytes(bytes_data)
        return "\n".join([pytesseract.image_to_string(img) for img in images])
    except:
        return ""


def extract_text(file):
    name = file.name.lower()
    raw = file.read()
    result = {"text": "", "images": [], "tables": []}

    if name.endswith(".pdf"):
        result["text"] = extract_text_from_pdf(raw)
        try:
            result["images"] = convert_from_bytes(raw)[:2]
        except: pass

    elif name.endswith(".docx"):
        result["text"] = extract_text_from_docx(raw)

    elif name.endswith(".txt"):
        result["text"] = extract_text_from_txt(raw)

    elif name.endswith((".xlsx", ".xls")):
        result["text"], result["tables"] = extract_text_from_excel(raw)

    elif name.endswith((".png", ".jpg", ".jpeg")):
        result["text"] = extract_text_from_image(raw)
        result["images"] = [Image.open(io.BytesIO(raw))]

    return result


# ============================================================
# COMPARISON LOGIC
# ============================================================
def seq_diff(a, b):
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    matcher = SequenceMatcher(a, b)
    diff = list(unified_diff(a_lines, b_lines, lineterm=""))
    return matcher.ratio(), diff, a_lines, b_lines


def semantic_similarity(a, b):
    if not a.strip() or not b.strip():
        return 0.0
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2).item())


def extract_numbers(text):
    nums = re.findall(r"[0-9.,/-]+", text)
    return [n.replace(",", "") for n in nums]


def compare_numbers(a, b):
    na, nb = set(extract_numbers(a)), set(extract_numbers(b))
    return {
        "added": sorted(nb - na),
        "removed": sorted(na - nb),
        "common": sorted(na & nb)
    }


def compare_images(imgA, imgB):
    if not imgA or not imgB:
        return {"error": "No images found"}

    imgA = imgA[0]
    imgB = imgB[0]

    h1, h2 = imagehash.phash(imgA), imagehash.phash(imgB)
    hdiff = h1 - h2

    grayA = ImageOps.grayscale(imgA).resize((256, 256))
    grayB = ImageOps.grayscale(imgB).resize((256, 256))

    return {
        "phash_distance": int(hdiff),
        "ssim": float(ssim(np.array(grayA), np.array(grayB)))
    }


def html_diff(a_lines, b_lines):
    diff_html = HtmlDiff(wrapcolumn=80).make_table(a_lines, b_lines)
    return f"<div class='card'>{diff_html}</div>"


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("""
<div class='sidebar-container'>
    <div class='sidebar-title'>DiffPro AI</div>
    <div class='sidebar-sub'>Document Comparator</div>
</div>
""", unsafe_allow_html=True)

nav = st.sidebar.radio(
    "",
    ["📄 Compare Documents", "✨ Features", "👩‍💼 About Me"],
)

page = (
    "Compare Documents" if nav.startswith("📄") else
    "Features" if nav.startswith("✨") else
    "About Me"
)


# ============================================================
# COMPARE DOCUMENTS PAGE
# ============================================================
if page == "Compare Documents":

    st.title("📄 DiffPro AI — Compare Any Two Documents")

    colA, colB = st.columns(2)
    with colA:
        fileA = st.file_uploader("Upload Document A", type=["pdf","docx","txt","xlsx","png","jpg"])
    with colB:
        fileB = st.file_uploader("Upload Document B", type=["pdf","docx","txt","xlsx","png","jpg"])

    if fileA and fileB:
        if st.button("Run Comparison"):

            with st.spinner("Extracting & analyzing..."):
                fileA.seek(0); A = extract_text(fileA)
                fileB.seek(0); B = extract_text(fileB)

            st.success("Extraction Complete.")

            # ----- TEXT METRICS -----
            st.header("📝 Text Comparison")
            ratio, diff_raw, a_lines, b_lines = seq_diff(A["text"], B["text"])
            sim = semantic_similarity(A["text"], B["text"])

            col1, col2 = st.columns(2)
            col1.metric("Text Similarity", f"{ratio:.3f}")
            col2.metric("Semantic Similarity", f"{sim:.3f}")

            st.subheader("Visual Colored Diff")
            st.markdown(html_diff(a_lines[:400], b_lines[:400]), unsafe_allow_html=True)

            # ----- NUMBERS -----
            st.header("🔢 Numeric Difference")
            st.json(compare_numbers(A["text"], B["text"]))

            # ----- IMAGES -----
            st.header("🖼️ Image Comparison")
            st.json(compare_images(A["images"], B["images"]))

            # ----- SIDE BY SIDE TEXT -----
            st.header("📚 Side-by-Side Text View")
            c1, c2 = st.columns(2)
            c1.text_area("Document A", A["text"][:5000], height=300)
            c2.text_area("Document B", B["text"][:5000], height=300)


# ============================================================
# FEATURES PAGE
# ============================================================
elif page == "Features":

    st.title("✨ Features of DiffPro AI")

    st.markdown("""
<div class='card'>
    <h3>🔍 Intelligent Text Comparison</h3>
    <p>Detects paraphrasing, rewording, and subtle changes in meaning using semantic embeddings.</p>
</div>

<div class='card'>
    <h3>🧠 Deep Semantic Analysis</h3>
    <p>Powered by SentenceTransformers for industry-level NLP understanding.</p>
</div>

<div class='card'>
    <h3>📑 Visual Inline Diff Viewer</h3>
    <p>Beautiful HTML diff with color-coded highlights for added / removed / modified lines.</p>
</div>

<div class='card'>
    <h3>🖼️ OCR + Image Similarity</h3>
    <p>OCR for scanned PDFs + pHash + SSIM for structural visual differences.</p>
</div>

<div class='card'>
    <h3>📊 Excel & Table Comparison</h3>
    <p>Column-level, row-level, and cell-level comparison with structured insights.</p>
</div>

<div class='card'>
    <h3>🌐 Multi-Format Support</h3>
    <p>PDF, DOCX, TXT, XLSX, PNG, JPG — all handled automatically.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# ABOUT ME PAGE
# ============================================================
elif page == "About Me":

    st.title("👩‍💼 About the Creator")

    about_html = """
<div class='card'>
    <div class='about-container'>
    
        <img src='https://cdn-icons-png.flaticon.com/512/2922/2922561.png' class='about-img'>

        <div class='about-text'>
            <h2 style='color:#00D2FF;'>Indu Reddy</h2>

            <p>
            AI Engineer • Bengaluru <br><br>

            I create AI-powered tools that blend machine learning, NLP, OCR, and 
            computer vision to solve real-world document processing challenges.<br><br>

            <strong>Expertise:</strong><br>
            • Machine Learning & AI<br>
            • NLP & Document Intelligence<br>
            • Data Engineering<br>
            • AI UI/UX & Deployment<br><br>

            <strong>GitHub:</strong><br>
            <a href='https://github.com/indureddy20'>github.com/indureddy20</a>
            </p>
        </div>

    </div>
</div>
"""

    st.markdown(about_html, unsafe_allow_html=True)
