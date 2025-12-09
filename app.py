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
# PAGE: FEATURES (replacement)
# ============================
elif page == "Features":
    st.title("✨ Features of DiffPro AI")

    # feature definitions (title, short desc, accent color)
    features = [
        ("AI Semantic Comparison", "Detects paraphrase & semantic drift using transformer embeddings.", "#00d2ff"),
        ("OCR for Scanned PDFs", "Extract text from scanned PDFs/images reliably with Tesseract OCR.", "#7c4dff"),
        ("Excel & Table Support", "Sheet & cell level summaries, quick table previews.", "#00bfa5"),
        ("Image Visual Diff", "Perceptual hashing & SSIM for image similarity and drift.", "#ff6b6b"),
        ("Modern UI Experience", "Polished glass cards, responsive layout, and animations.", "#ffd93d"),
        ("Downloadable Reports", "Export structured JSON reports for auditing & automation.", "#4caf50"),
        ("100% Free Forever", "Free to use, self-host or deploy to Streamlit Cloud.", "#ff4081"),
    ]

    # build a responsive 3-column grid of cards (inline styles so color/visibility are stable)
    cols = st.columns(3, gap="large")
    for i, (title, desc, accent) in enumerate(features):
        col = cols[i % 3]
        with col:
            # inline styling ensures the text stands out regardless of global CSS
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
                    border-radius:16px; padding:28px; min-height:160px;
                    display:flex; flex-direction:column; justify-content:center;
                    box-shadow:0 12px 36px rgba(2,8,30,0.6);
                    border:1px solid rgba(255,255,255,0.04);
                ">
                    <div style="font-weight:800; font-size:1.2rem; color: #ffffff; margin-bottom:10px;">
                        <span style="display:inline-block; width:12px; height:12px; background:{accent}; border-radius:3px; margin-right:10px;"></span>
                        {title}
                    </div>
                    <div style="color: rgba(220,220,255,0.9); font-size:0.95rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # small spacer & optional explanation
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ============================
# PAGE: ABOUT ME (replacement)
# ============================
elif page == "About Me":
    st.title("👩‍💼 About the Creator")

    # Use columns (left image, right text) — avoids long HTML strings that can leak
    col_img, col_text = st.columns([1, 2], gap="large")

    with col_img:
        # corporate illustrated avatar (Option A)
        avatar_url = "https://cdn-icons-png.flaticon.com/512/2922/2922561.png"
        st.image(avatar_url, width=240, caption="Indu Reddy", use_column_width=False)

    with col_text:
        # Use markdown with minimal inline HTML; closed string ensures no leakage
        st.markdown(
            """
            <div style="background: rgba(255,255,255,0.02); padding:18px; border-radius:12px; border:1px solid rgba(255,255,255,0.03);">
              <h2 style="color:#00D2FF; margin-top:0; margin-bottom:6px;">Indu Reddy</h2>
              <p style="color:#E8E8FF; font-size:1.05rem; line-height:1.6; margin-top:0;">
                <strong>AI Engineer • Bengaluru</strong><br><br>
                I design advanced AI-powered tools that solve real-world problems using Machine Learning, NLP, OCR, and Computer Vision.
                DiffPro AI compares PDFs, Word files, Excel sheets, and images using semantic analysis, OCR extraction, and visual similarity.<br><br>
                <strong>Expertise:</strong><br>
                • Artificial Intelligence & Machine Learning<br>
                • NLP & Document Intelligence<br>
                • OCR, Embeddings & Vision Models<br>
                • Deployment & UI Engineering<br><br>
                <strong>GitHub:</strong> <a href="https://github.com/indureddy20" style="color:#7C4DFF;">github.com/indureddy20</a>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # small spacer
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

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
