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

# NLP Model & NLTK
nltk.download('punkt', quiet=True)

@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
model = get_model()


# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown("""
<style>

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0A0A1F 0%, #1A1A3A 50%, #2D1B69 100%);
    color: #E8E8FF;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141428 0%, #0A0A15 100%);
    padding-top: 20px;
}
.sidebar-title { text-align:center; }
.sidebar-title h2 { color:#00D2FF; font-size:1.9rem; margin-bottom:2px; }
.sidebar-title p { color:#AAB4FF; font-size:1rem; }

.stSidebar label { color:#E8E8FF !important; font-size:1.15rem !important; }

/* Button */
.stButton > button {
    background: linear-gradient(45deg, #00D2FF, #7C4DFF);
    color:white !important;
    padding:14px 36px;
    border-radius:40px;
    border:none;
    font-size:1.15rem;
    font-weight:600;
    transition:0.25s;
}
.stButton > button:hover { transform:translateY(-4px); }

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    padding:1.8rem;
    border-radius:14px;
    border:1px solid rgba(255,255,255,0.08);
    margin-bottom:25px;
    backdrop-filter:blur(10px);
}

/* Diff Table */
table.diff { width:100%; border-collapse:collapse; font-family:monospace; }
table.diff th { background:rgba(255,255,255,0.06); }
.diff_add { background:rgba(76,175,80,0.28) !important; }
.diff_sub { background:rgba(244,67,54,0.28) !important; }
.diff_chg { background:rgba(255,202,40,0.28) !important; }

/* About Page */
.about-container { display:flex; gap:2rem; align-items:center; }
.about-img {
    width:180px; height:180px; border-radius:50%;
    border:3px solid #00D2FF;
    box-shadow:0 0 18px rgba(0,210,255,0.4);
}
.about-text { font-size:1.2rem; line-height:1.7; color:#EEE; }
.about-text a { color:#7C4DFF; font-weight:700; }

</style>
""", unsafe_allow_html=True)


# ============================================================
# EXTRACTION UTILITIES
# ============================================================
def extract_text_from_docx(raw):
    doc = docx.Document(io.BytesIO(raw))
    return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])


def extract_text_from_txt(raw):
    try:
        return raw.decode("utf-8", errors="ignore")
    except:
        return str(raw)


def extract_text_from_image(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return pytesseract.image_to_string(img)


def extract_text_from_excel(raw):
    try:
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        text_parts = []
        table_data = []

        for name, df in sheets.items():
            text_parts.append(f"Sheet: {name}\nColumns: {list(df.columns)}\n{df.head()}")
            table_data.append((name, df))

        return "\n".join(text_parts), table_data
    except:
        return "", []


def extract_text_from_pdf(raw):
    try:
        pdf = pdfplumber.open(io.BytesIO(raw))
        text = "\n".join([pg.extract_text() or "" for pg in pdf.pages])
        if text.strip():
            return text
    except:
        pass

    try:
        images = convert_from_bytes(raw)
        return "\n".join([pytesseract.image_to_string(img) for img in images])
    except:
        return ""


def extract_text(file):
    name = file.name.lower()
    raw = file.read()
    out = {"text": "", "images": [], "tables": []}

    if name.endswith(".pdf"):
        out["text"] = extract_text_from_pdf(raw)
        try: out["images"] = convert_from_bytes(raw)[:2]
        except: pass

    elif name.endswith(".docx"):
        out["text"] = extract_text_from_docx(raw)

    elif name.endswith(".txt"):
        out["text"] = extract_text_from_txt(raw)

    elif name.endswith((".xlsx", ".xls")):
        out["text"], out["tables"] = extract_text_from_excel(raw)

    elif name.endswith((".png", ".jpg", ".jpeg")):
        out["text"] = extract_text_from_image(raw)
        out["images"] = [Image.open(io.BytesIO(raw))]

    return out


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
    e1, e2 = model.encode(a, convert_to_tensor=True), model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(e1, e2))


def extract_numbers(t):
    nums = re.findall(r"[0-9.,/-]+", t)
    return [n.replace(",", "") for n in nums]


def compare_numbers(a, b):
    A, B = set(extract_numbers(a)), set(extract_numbers(b))
    return {"added": sorted(B-A), "removed": sorted(A-B), "common": sorted(A & B)}


def compare_images(imgA, imgB):
    if not imgA or not imgB:
        return {"error": "No images found"}

    A, B = imgA[0], imgB[0]
    h1, h2 = imagehash.phash(A), imagehash.phash(B)
    phash_dist = h1 - h2

    g1 = ImageOps.grayscale(A).resize((256,256))
    g2 = ImageOps.grayscale(B).resize((256,256))
    s = float(ssim(np.array(g1), np.array(g2)))

    return {"phash_distance": int(phash_dist), "ssim": s}


def html_diff(a_lines, b_lines):
    return "<div class='card'>" + HtmlDiff(wrapcolumn=80).make_table(a_lines, b_lines) + "</div>"


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("""
<div class='sidebar-title'>
    <h2>DiffPro AI</h2>
    <p>Document Comparator</p>
</div>
""", unsafe_allow_html=True)

nav = st.sidebar.radio(
    "",
    ["📄 Compare Documents", "✨ Features", "👩‍💼 About Me"]
)

page = (
    "Compare Documents" if nav.startswith("📄") else
    "Features" if nav.startswith("✨") else
    "About Me"
)


# ============================================================
# PAGE: COMPARE DOCUMENTS
# ============================================================
if page == "Compare Documents":

    st.title("📄 Compare Documents")

    colA, colB = st.columns(2)
    with colA:
        fileA = st.file_uploader("Upload Document A", type=["pdf","docx","txt","xlsx","png","jpg"])
    with colB:
        fileB = st.file_uploader("Upload Document B", type=["pdf","docx","txt","xlsx","png","jpg"])

    if fileA and fileB:
        if st.button("Run Comparison"):

            with st.spinner("Extracting & comparing..."):
                fileA.seek(0); A = extract_text(fileA)
                fileB.seek(0); B = extract_text(fileB)

            st.success("Extraction complete!")

            # TEXT METRICS
            st.header("📝 Text Comparison")
            ratio, _, a_lines, b_lines = seq_diff(A["text"], B["text"])
            sem = semantic_similarity(A["text"], B["text"])

            c1, c2 = st.columns(2)
            c1.metric("Text Similarity", f"{ratio:.3f}")
            c2.metric("Semantic Similarity", f"{sem:.3f}")

            st.subheader("Visual Inline Diff Viewer")
            st.markdown(html_diff(a_lines[:400], b_lines[:400]), unsafe_allow_html=True)

            # NUMERIC DIFFERENCES
            st.header("🔢 Numeric Field Comparison")
            st.json(compare_numbers(A["text"], B["text"]))

            # IMAGES
            st.header("🖼️ Image Comparison")
            st.json(compare_images(A["images"], B["images"]))

            # SIDE-BY-SIDE
            st.header("📚 Side-by-Side Text")
            c1, c2 = st.columns(2)
            c1.text_area("Document A", A["text"][:5000], height=300)
            c2.text_area("Document B", B["text"][:5000], height=300)


# ============================================================
# PAGE: FEATURES
# ============================================================
elif page == "Features":

    st.title("✨ Features of DiffPro AI")

    st.markdown("""
<div class='card'>
    <h3>🔍 Intelligent Text Comparison</h3>
    <p>
        DiffPro AI detects exact text changes, missing sections, modified lines,
        paraphrased writing, and content rearrangements.
    </p>
</div>

<div class='card'>
    <h3>🧠 Deep Semantic Analysis</h3>
    <p>
        Using state-of-the-art transformer embeddings, DiffPro determines how similar
        the meaning of two documents is — even when rewritten.
    </p>
</div>

<div class='card'>
    <h3>📑 Visual Inline Diff Viewer</h3>
    <p>
        A clean HTML diff with color-coded highlights:<br>
        <span style='color:#4CAF50;'>🟩 Added</span><br>
        <span style='color:#FF5252;'>🟥 Removed</span><br>
        <span style='color:#FFCA28;'>🟨 Modified</span><br>
        Makes reviewing changes extremely simple.
    </p>
</div>

<div class='card'>
    <h3>🖼️ OCR + Image Comparison</h3>
    <p>
        Extracts text from scanned PDFs, and compares images using pHash and SSIM — 
        great for documents containing screenshots or scanned pages.
    </p>
</div>

<div class='card'>
    <h3>📊 Excel & Table Comparison</h3>
    <p>
        DiffPro analyzes sheet differences, column changes, row drift, and mismatched
        cells. Very useful for financial or audit documents.
    </p>
</div>

<div class='card'>
    <h3>📤 JSON Comparison Report</h3>
    <p>
        Provides a detailed JSON containing text, semantic, numeric, table, and image
        comparison results — ideal for automation and audits.
    </p>
</div>

<div class='card'>
    <h3>🌐 Multi-format Document Support</h3>
    <p>
        Works with PDF, DOCX, TXT, XLSX, PNG, JPG — no configuration needed.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE: ABOUT ME
# ============================================================
elif page == "About Me":

    st.title("👩‍💼 About the Creator")

    st.markdown("""
<div class='card'>
    <div class='about-container'>

        <img src='https://cdn-icons-png.flaticon.com/512/2922/2922561.png' class='about-img'>

        <div class='about-text'>
            <h2 style='color:#00D2FF;'>Indu Reddy</h2>

            <p>
            AI Engineer • Bengaluru <br><br>

            I build intelligent applications using Machine Learning, NLP,
            Computer Vision, and interactive UI systems.  
            DiffPro AI is designed to help individuals, businesses, and analysts
            compare documents with precision and clarity.<br><br>

            <strong>Expertise:</strong><br>
            • Artificial Intelligence & Machine Learning<br>
            • NLP & Document Intelligence<br>
            • OCR, Embeddings & Vision Models<br>
            • Data Engineering & Deployment<br><br>

            <strong>GitHub:</strong><br>
            <a href='https://github.com/indureddy20'>github.com/indureddy20</a>
            </p>

        </div>

    </div>
</div>
""", unsafe_allow_html=True)
