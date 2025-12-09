# ============================
# DiffPro AI – Restored Original UI (Polished + Bug-free)
# ============================
import streamlit as st
import io, re, json, base64
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

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="DiffPro AI - Document Comparator",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# NLTK
nltk.download("punkt", quiet=True)


# ============================
# LOAD AI MODEL
# ============================
@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = get_model()

# ============================
# GLOBAL CSS – RESTORED UI THEME
# ============================
st.markdown(
    """
<style>
html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg,#0A0A1F,#1A1A3A,#2D1B69);
    color:#E8E8FF;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#131324,#0C0C18);
}
.sidebar-title { text-align:center; padding:10px 5px; }
.sidebar-title h2 { color:#00D2FF; margin:0; font-weight:800; }
.sidebar-title p { color:#AAB4FF; margin-top:4px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(45deg,#00D2FF,#7C4DFF);
    padding: 12px 30px;
    border-radius: 50px;
    border:none;
    font-size:1.1rem;
    font-weight:700;
    color:white !important;
    box-shadow:0 8px 20px rgba(0,0,0,0.4);
}
.stButton > button:hover { transform: translateY(-3px); }

/* Card styling */
.card {
    background: rgba(255,255,255,0.05);
    border-radius:18px;
    padding:24px;
    margin-bottom:20px;
    border:1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(12px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
}

/* Feature cards */
.feature-card {
    background: rgba(40,40,80,0.35);
    padding:26px;
    border-radius:22px;
    text-align:center;
    transition:0.4s;
    border:1px solid rgba(90,90,255,0.28);
    box-shadow:0 12px 30px rgba(0,0,0,0.5);
}
.feature-card:hover {
    transform: translateY(-10px);
    background: rgba(80,80,200,0.4);
}

/* About section */
.about-container {
    display:flex;
    align-items:center;
    gap:2rem;
}
.about-img {
    width:200px;
    height:200px;
    border-radius:50%;
    border:3px solid #00D2FF;
    box-shadow:0 0 25px rgba(0,210,255,0.45);
}
.about-text {
    font-size:1.2rem;
    line-height:1.7;
    color:#E0E0FF;
}

/* Diff table colors */
.diff_add { background: rgba(76,175,80,0.28) !important; }
.diff_sub { background: rgba(244,67,54,0.28) !important; }
.diff_chg { background: rgba(255,202,40,0.28) !important; }

/* Footer */
.footer {
    text-align:center;
    margin-top:40px;
    padding:14px;
    opacity:0.75;
    font-size:0.95rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================
# EXTRACTION FUNCTIONS
# ============================
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
        out = []
        tables = []
        for name, df in sheets.items():
            tables.append((name, df))
            out.append(f"Sheet: {name}\nColumns: {list(df.columns)}\n{df.head().to_string()}")
        return "\n".join(out), tables
    except:
        return "", []


def extract_text_from_pdf(raw):
    try:
        pdf = pdfplumber.open(io.BytesIO(raw))
        pages = [pg.extract_text() or "" for pg in pdf.pages]
        text = "\n".join(pages)
        if text.strip():
            return text
    except:
        pass

    try:
        imgs = convert_from_bytes(raw)
        return "\n".join([pytesseract.image_to_string(img) for img in imgs])
    except:
        return ""


def extract_text(file):
    name = file.name.lower()
    data = file.read()

    result = {"text": "", "images": [], "tables": []}

    if name.endswith(".pdf"):
        result["text"] = extract_text_from_pdf(data)
        try:
            result["images"] = convert_from_bytes(data)[:2]
        except:
            pass

    elif name.endswith(".docx"):
        result["text"] = extract_text_from_docx(data)

    elif name.endswith(".txt"):
        result["text"] = extract_text_from_txt(data)

    elif name.endswith(".xlsx"):
        t, tables = extract_text_from_excel(data)
        result["text"] = t
        result["tables"] = tables

    elif name.endswith((".png", ".jpg", ".jpeg")):
        result["text"] = extract_text_from_image(data)
        result["images"] = [Image.open(io.BytesIO(data))]

    return result


# ============================
# COMPARISON FUNCTIONS
# ============================
def seq_diff(a, b):
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    matcher = SequenceMatcher(None, a, b)
    diff = list(unified_diff(a_lines, b_lines, lineterm=""))
    return matcher.ratio(), diff, a_lines, b_lines


def semantic_similarity(a, b):
    if not a.strip() or not b.strip():
        return 0.0
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))


def extract_numbers(text):
    return re.findall(r"[0-9.,/-]+", text)


def compare_numbers(a, b):
    na = set(extract_numbers(a))
    nb = set(extract_numbers(b))
    return {"added": sorted(nb - na), "removed": sorted(na - nb), "common": sorted(na & nb)}


def compare_images(imgA, imgB):
    if not imgA or not imgB:
        return {"error": "No images available"}

    A = imgA[0]
    B = imgB[0]

    try:
        ph = int(imagehash.phash(A) - imagehash.phash(B))
    except:
        ph = None

    try:
        ga = ImageOps.grayscale(A).resize((256, 256))
        gb = ImageOps.grayscale(B).resize((256, 256))
        ssim_val = float(ssim(np.array(ga), np.array(gb)))
    except:
        ssim_val = None

    return {"phash_distance": ph, "ssim": ssim_val}


def html_diff(a_lines, b_lines):
    return HtmlDiff(wrapcolumn=80).make_table(a_lines, b_lines)


# ============================
# SIDEBAR NAVIGATION
# ============================
st.sidebar.markdown(
    """
<div class='sidebar-title'>
    <h2>DiffPro AI</h2>
    <p>Document Comparator</p>
</div>
""",
    unsafe_allow_html=True,
)

nav = st.sidebar.radio("", ["📄 Compare Documents", "✨ Features", "👩‍💼 About Me"])
page = "Compare Documents" if nav.startswith("📄") else "Features" if nav.startswith("✨") else "About Me"


# ============================
# PAGE: COMPARE DOCUMENTS
# ============================
if page == "Compare Documents":
    st.title("📄 DiffPro AI — Compare Any Two Documents")

    colA, colB = st.columns(2)
    with colA:
        fileA = st.file_uploader("Upload Document A", type=["pdf", "docx", "txt", "xlsx", "png", "jpg"])
    with colB:
        fileB = st.file_uploader("Upload Document B", type=["pdf", "docx", "txt", "xlsx", "png", "jpg"])

    if fileA and fileB and st.button("Run Comparison"):
        fileA.seek(0)
        A = extract_text(fileA)
        fileB.seek(0)
        B = extract_text(fileB)

        st.success("Extraction complete.")

        # ===== Text comparison =====
        st.header("📝 Text Comparison")
        ratio, diff_raw, a_lines, b_lines = seq_diff(A["text"], B["text"])
        sem = semantic_similarity(A["text"], B["text"])

        c1, c2 = st.columns(2)
        c1.metric("Text Similarity", f"{ratio:.3f}")
        c2.metric("Semantic Similarity", f"{sem:.3f}")

        st.subheader("Inline Diff Viewer")
        st.markdown(html_diff(a_lines[:300], b_lines[:300]), unsafe_allow_html=True)

        # ===== Numeric comparison =====
        st.header("🔢 Numeric Comparison")
        st.json(compare_numbers(A["text"], B["text"]))

        # ===== Image comparison =====
        st.header("🖼 Image Comparison")
        st.json(compare_images(A["images"], B["images"]))

        # ===== Raw text side-by-side =====
        col1, col2 = st.columns(2)
        col1.text_area("Document A", A["text"][:5000], height=300)
        col2.text_area("Document B", B["text"][:5000], height=300)


# ============================
# PAGE: FEATURES — RESTORED UI
# ============================
elif page == "Features":

    st.title("✨ Why Choose DiffPro AI?")

    # 7 feature cards (restored)
    features = [
        ("AI Semantic Comparison", "fa-solid fa-brain fa-beat", "#00d2ff"),
        ("OCR for Scanned PDFs", "fa-solid fa-eye fa-flip", "#7c4dff"),
        ("Excel & Table Support", "fa-solid fa-table fa-bounce", "#00bfa5"),
        ("Image Visual Diff", "fa-solid fa-images fa-shake", "#ff6b6b"),
        ("Modern UI Experience", "fa-solid fa-gem fa-spin", "#ffd93d"),
        ("Downloadable Reports", "fa-solid fa-download fa-bounce", "#4caf50"),
        ("100% Free Forever", "fa-solid fa-heart fa-beat-fade", "#ff4081"),
    ]

    cols = st.columns(3)
    idx = 0
    for title, icon, color in features:
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class='feature-card'>
                    <h2 style='font-size:3rem; margin:0;'>
                        <i class="{icon}" style="color:{color};"></i>
                    </h2>
                    <h4 style='margin-top:1rem;'>{title}</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
        idx += 1


# ============================
# PAGE: ABOUT ME — RESTORED
# ============================
elif page == "About Me":

    st.title("👩‍💼 About the Creator")

    st.markdown(
        """
<div class='card'>
    <div class='about-container'>

        <img src='https://cdn-icons-png.flaticon.com/512/2922/2922561.png' class='about-img'>

        <div class='about-text'>
            <h2 style='color:#00D2FF;'>Indu Reddy</h2>

            <p>
            AI Engineer • Bengaluru <br><br>
            I design advanced AI-powered tools that solve real-world problems using
            Machine Learning, NLP, OCR, and Computer Vision.<br><br>

            DiffPro AI compares PDFs, Word files, Excel sheets, and images using
            semantic analysis, OCR extraction, and visual similarity.<br><br>

            <strong>Expertise:</strong><br>
            • Artificial Intelligence & Machine Learning<br>
            • NLP & Document Intelligence<br>
            • OCR, Embeddings & Vision Models<br>
            • Deployment & UI Engineering<br><br>

            <strong>GitHub:</strong><br>
            <a href='https://github.com/indureddy20' style='color:#7C4DFF;'>github.com/indureddy20</a>
            </p>
        </div>

    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================
# FOOTER
# ============================
st.markdown(
    """
<div class='footer'>
    Built with ❤️ and passion
</div>
""",
    unsafe_allow_html=True,
)
