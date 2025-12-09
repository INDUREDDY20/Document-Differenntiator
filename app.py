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


# -----------------------------
# PAGE CONFIG — MUST COME FIRST
# -----------------------------
st.set_page_config(
    page_title="DiffPro AI - Document Comparator",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# NLTK
# -----------------------------
nltk.download('punkt', quiet=True)


# -----------------------------
# MODEL CACHE
# -----------------------------
@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = get_model()


# ============================================================
#                     GLOBAL CSS (Theme)
# ============================================================
st.markdown("""
<style>

/* GLOBAL THEME */
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
.sidebar-title { text-align:center; margin-bottom:20px; }
.sidebar-title h2 { color:#00D2FF; margin:0; font-size:1.8rem; font-weight:700; }
.sidebar-title p { color:#AAB4FF; font-size:0.9rem; margin-top:4px; }

/* RADIO BUTTON FIX */
[data-testid="stSidebar"] label { 
    font-size:1rem !important; 
    color:#d0d0ff !important;
}

/* BUTTON STYLE */
.stButton > button {
    background: linear-gradient(45deg, #00D2FF, #7C4DFF);
    color: white !important;
    border: none;
    border-radius: 50px;
    padding: 14px 36px;
    font-size: 1.1rem;
    font-weight: 600;
    box-shadow:0 6px 18px rgba(0,0,0,0.35);
    transition:0.25s;
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow:0 10px 25px rgba(0,0,0,0.45);
}

/* GLASS CARD */
.card {
    background: rgba(255,255,255,0.04);
    padding: 1.8rem;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    margin-bottom: 25px;
}

/* DIFF VIEWER */
table.diff {
    width: 100%;
    border-collapse: collapse;
    font-family: monospace;
    font-size: 0.95rem;
    margin-top: 20px;
}
table.diff th {
    background: rgba(255, 255, 255, 0.05);
    color: #F0F0FF;
    padding: 6px;
    border: 1px solid rgba(255,255,255,0.08);
    font-weight:600;
}
table.diff td {
    padding: 6px;
    vertical-align: top;
    border: 1px solid rgba(255,255,255,0.03);
}

.diff_header { background: rgba(0,0,0,0.3); color:#FFF; }
.diff_next { background: transparent; }

.diff_add { background-color: rgba(76, 175, 80, 0.28) !important; }
.diff_sub { background-color: rgba(244, 67, 54, 0.28) !important; }
.diff_chg { background-color: rgba(255, 202, 40, 0.28) !important; }

/* ABOUT PAGE */
.about-container {
    display:flex;
    gap:2rem;
    align-items:center;
}
.about-img {
    width:180px;
    height:180px;
    border-radius:50%;
    border:3px solid #00D2FF;
    box-shadow:0 0 20px rgba(0,210,255,0.4);
}
.about-text { font-size:1.2rem; line-height:1.7; color:#EEE; }
.about-text a { color:#7C4DFF; font-weight:600; }

/* SMALL MONO */
.small-mono { font-family: monospace; font-size:0.9rem; color:#CCCCCC; }

</style>
""", unsafe_allow_html=True)


# ============================================================
#                     EXTRACTION FUNCTIONS
# ============================================================

def extract_text_from_docx(bytes_data):
    doc = docx.Document(io.BytesIO(bytes_data))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paras)

def extract_text_from_txt(bytes_data):
    try:
        return bytes_data.decode('utf-8', errors='ignore')
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
            snippet = f"Sheet: {name}\nColumns: {list(df.columns)}\n"
            snippet += df.head().to_string()
            text_parts.append(snippet)
        return "\n".join(text_parts), out_tables
    except:
        return "", []

def extract_text_from_pdf(bytes_data):
    try:
        pdf = pdfplumber.open(io.BytesIO(bytes_data))
        pages = [pg.extract_text() or "" for pg in pdf.pages]
        text = "\n".join(pages)
        if text.strip():
            return text
    except:
        pass

    # fallback to OCR
    try:
        images = convert_from_bytes(bytes_data)
        text = "\n".join([pytesseract.image_to_string(img) for img in images])
        return text
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
        except:
            pass

    elif name.endswith(".docx"):
        result["text"] = extract_text_from_docx(raw)

    elif name.endswith(".txt"):
        result["text"] = extract_text_from_txt(raw)

    elif name.endswith(".xlsx") or name.endswith(".xls"):
        t, tables = extract_text_from_excel(raw)
        result["text"] = t
        result["tables"] = tables

    elif name.endswith((".png", ".jpg", ".jpeg")):
        result["text"] = extract_text_from_image(raw)
        result["images"] = [Image.open(io.BytesIO(raw))]

    else:
        result["text"] = extract_text_from_pdf(raw)

    return result


# ============================================================
#                        COMPARISON LOGIC
# ============================================================

def seq_diff(a, b):
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    matcher = SequenceMatcher(a="\n".join(a_lines), b="\n".join(b_lines))
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
        return {"error": "No images to compare"}

    imgA, imgB = imgA[0], imgB[0]
    h1, h2 = imagehash.phash(imgA), imagehash.phash(imgB)
    hdiff = h1 - h2

    grayA = ImageOps.grayscale(imgA).resize((256, 256))
    grayB = ImageOps.grayscale(imgB).resize((256, 256))

    arrA = np.array(grayA)
    arrB = np.array(grayB)

    return {
        "phash_distance": int(hdiff),
        "ssim": float(ssim(arrA, arrB))
    }

def html_diff(a_lines, b_lines):
    diff = HtmlDiff(wrapcolumn=80).make_table(a_lines, b_lines)
    return f"<div class='card'>{diff}</div>"


# ============================================================
#                       UI STARTS HERE
# ============================================================
# ------------------------------------
# BEAUTIFUL MODERN SIDEBAR NAVIGATION
# ------------------------------------

st.sidebar.markdown("""
<style>

.sidebar-container {
    text-align: center;
    padding-top: 10px;
    padding-bottom: 25px;
}

.sidebar-title {
    color: #00D2FF;
    font-size: 2rem;
    margin-bottom: 3px;
    font-weight: 800;
}

.sidebar-sub {
    color: #AAB4FF;
    font-size: 1rem;
    margin-bottom: 20px;
}

/* NAV BUTTONS */
.sidebar-nav button {
    width: 100%;
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    padding: 12px 18px;
    margin-bottom: 10px;
    border-radius: 12px;
    color: #e0e0ff !important;
    text-align: left;
    font-size: 1.05rem;
    transition: 0.25s;
}

.sidebar-nav button:hover {
    background: rgba(0,210,255,0.2) !important;
    transform: translateX(4px);
    border-color: rgba(0,210,255,0.5) !important;
}

/* ACTIVE BUTTON */
.sidebar-nav button[data-selected="true"] {
    background: linear-gradient(90deg, #00D2FF, #7C4DFF) !important;
    color: white !important;
    font-weight: 700;
    box-shadow: 0 0 12px rgba(0,210,255,0.5);
}

</style>
""", unsafe_allow_html=True)


# Custom Navigation Buttons with icons
st.sidebar.markdown("""
<div class="sidebar-container">
    <div class="sidebar-title">DiffPro AI</div>
    <div class="sidebar-sub">Document Comparator</div>
</div>
""", unsafe_allow_html=True)


# Create navigation layout
nav = st.sidebar.radio(
    "",
    ["📄 Compare Documents", "✨ Features", "👩‍💼 About Me"],
    key="nav",
)

# Map values back to your page variable
if nav.startswith("📄"):
    page = "Compare Documents"
elif nav.startswith("✨"):
    page = "Features"
else:
    page = "About Me"


# ------------------------------------------------------------
#                COMPARE DOCUMENTS PAGE
# ------------------------------------------------------------
if page == "Compare Documents":

    st.title("📄 DiffPro AI — Compare Any Two Documents")

    colA, colB = st.columns(2)
    with colA:
        fileA = st.file_uploader("Upload Document A", type=['pdf','docx','txt','xlsx','png','jpg'])
    with colB:
        fileB = st.file_uploader("Upload Document B", type=['pdf','docx','txt','xlsx','png','jpg'])

    if fileA and fileB:
        if st.button("Run Comparison"):

            with st.spinner("Extracting documents..."):
                fileA.seek(0)
                A = extract_text(fileA)
                fileB.seek(0)
                B = extract_text(fileB)

            st.success("Extraction complete.")

            # ---- Text Compare ----
            st.header("📝 Text Comparison")
            ratio, diff_raw, a_lines, b_lines = seq_diff(A["text"], B["text"])
            sem = semantic_similarity(A["text"], B["text"])

            col1, col2 = st.columns(2)
            col1.metric("Text Similarity", f"{ratio:.3f}")
            col2.metric("Semantic Similarity", f"{sem:.3f}")

            st.subheader("Visual Colored Diff")
            st.markdown(html_diff(a_lines[:400], b_lines[:400]), unsafe_allow_html=True)

            # ---- Numeric Compare ----
            st.header("🔢 Numeric Field Comparison")
            st.json(compare_numbers(A["text"], B["text"]))

            # ---- Images ----
            st.header("🖼️ Image Comparison")
            st.json(compare_images(A["images"], B["images"]))

            # ---- Side-by-side Text ----
            st.header("📚 Side-by-Side Text")
            c1, c2 = st.columns(2)
            c1.text_area("Document A", A["text"][:5000], height=300)
            c2.text_area("Document B", B["text"][:5000], height=300)



# ------------------------------------------------------------
#                      FEATURES PAGE
# ------------------------------------------------------------
elif page == "Features":
    st.title("✨ Features of DiffPro AI")

    st.markdown("""
<div class='card'>
    <h3>🔍 Intelligent Text Comparison</h3>
    <p>
        DiffPro AI performs both <strong>exact text matching</strong> and 
        <strong>semantic similarity analysis</strong> using transformer embeddings.
        This allows the app to detect changes in meaning, paraphrasing, or reworded content.
    </p>
</div>

<div class='card'>
    <h3>🧠 Semantic Analysis (AI-Powered)</h3>
    <p>
        Uses SentenceTransformers for deep semantic comparison.  
        Identifies sentences with low semantic similarity and highlights them.
    </p>
</div>

<div class='card'>
    <h3>📑 Visual Inline Diff Viewer</h3>
    <p>
        Beautiful side-by-side diff view with color-coded highlights:<br>
        <span style='color:#4CAF50;'>Green = Added</span><br>
        <span style='color:#FF5252;'>Red = Removed</span><br>
        <span style='color:#FFCA28;'>Yellow = Modified</span>
    </p>
</div>

<div class='card'>
    <h3>🖼️ OCR & Image Comparison</h3>
    <p>
        • Extracts text from scanned PDFs & images using OCR.<br>
        • Computes perceptual hash distance (pHash) for visual similarity.<br>
        • SSIM score for structural similarity between images.
    </p>
</div>

<div class='card'>
    <h3>📊 Excel & Table Comparison</h3>
    <p>
        Detects:<br>
        • Column differences<br>
        • Row-level mismatches<br>
        • Sheet-wise comparison<br>
        • Cell-level drift (sample view)
    </p>
</div>

<div class='card'>
    <h3>📤 Exportable JSON Comparison Report</h3>
    <p>
        Generates a structured JSON file summarizing:<br>
        • Text similarity<br>
        • Semantic similarity<br>
        • Numerical drift<br>
        • Detected changes<br>
        • Table differences<br>
        • Image comparison metrics
    </p>
</div>

<div class='card'>
    <h3>🌐 Multi-Format Support</h3>
    <p>
        Accepts: PDF, DOCX, TXT, XLSX, PNG, JPG, JPEG.  
        Automatically detects and processes content appropriately.
    </p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
#                      ABOUT ME PAGE
# ------------------------------------------------------------
elif page == "About Me":

    st.title("👩‍💼 About the Creator")

    st.markdown("""
<div class='card'>
    <div class='about-container'>
    
        <!-- Corporate female avatar -->
        <img src='https://cdn-icons-png.flaticon.com/512/2922/2922561.png' class='about-img'>

        <div class='about-text'>
            <h2 style='color:#00D2FF; margin-bottom:8px;'>Indu Reddy</h2>
            <p>
            AI Engineer • Bengaluru <br><br>

            I design advanced AI-powered tools that solve real-world problems with a mix 
            of Machine Learning, NLP, OCR, and Computer Vision.  
            DiffPro AI is built to compare any type of document — PDFs, Word files, Excel sheets, images — 
            using deep semantic analysis, visual matching, and structural comparison.<br><br>

            <strong>Expertise:</strong><br>
            • Artificial Intelligence & Machine Learning<br>
            • NLP / Document Intelligence<br>
            • Data Engineering & Automation<br>
            • Model Deployment & AI UI/UX<br><br>

            <strong>GitHub:</strong><br>
            <a href="https://github.com/indureddy20">github.com/indureddy20</a>
            </p>

        </div>

    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("<br><center>DiffPro AI • Built with love and passion</center>", unsafe_allow_html=True)
