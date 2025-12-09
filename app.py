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


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="DiffPro AI - Document Comparator",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NLTK
nltk.download('punkt', quiet=True)


# ============================================================
# LOAD AI MODEL
# ============================================================
@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

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

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #131324 0%, #0C0C18 100%);
}
.sidebar-title { text-align:center; }
.sidebar-title h2 { color:#00D2FF; margin:0; font-size:1.8rem; }
.sidebar-title p { color:#AAB4FF; margin-top:4px; }

/* BUTTONS */
.stButton > button {
    background: linear-gradient(45deg, #00D2FF, #7C4DFF);
    padding: 14px 38px;
    border-radius: 40px;
    font-size: 1.1rem;
    font-weight: 600;
    border: none;
    color: white;
}
.stButton > button:hover { transform: translateY(-3px); }

/* GLASS CARD */
.card {
    background: rgba(255,255,255,0.05);
    padding: 1.8rem;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 25px;
    backdrop-filter: blur(10px);
}

/* DIFF TABLE */
table.diff { width:100%; border-collapse: collapse; font-family: monospace; }
table.diff th {
    background: rgba(255,255,255,0.1);
    padding: 6px; border:1px solid rgba(255,255,255,0.08);
}
table.diff td {
    padding: 6px; border:1px solid rgba(255,255,255,0.05);
}
.diff_add { background: rgba(76,175,80,0.3) !important; }
.diff_sub { background: rgba(244,67,54,0.3) !important; }
.diff_chg { background: rgba(255,202,40,0.3) !important; }

/* ABOUT PAGE */
.about-container { display:flex; align-items:center; gap:2rem; }
.about-img {
    width:180px; height:180px; border-radius:50%;
    border:3px solid #00D2FF;
    box-shadow:0 0 20px rgba(0,210,255,0.4);
}
.about-text { font-size:1.2rem; line-height:1.7; }

</style>
""", unsafe_allow_html=True)


# ============================================================
# EXTRACTION FUNCTIONS
# ============================================================
def extract_text_from_docx(data):
    d = docx.Document(io.BytesIO(data))
    return "\n".join([p.text for p in d.paragraphs if p.text.strip()])

def extract_text_from_txt(data):
    try: return data.decode("utf-8", errors="ignore")
    except: return str(data)

def extract_text_from_image(data):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return pytesseract.image_to_string(img)

def extract_text_from_excel(data):
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        out = []
        tables = []
        for name, df in sheets.items():
            tables.append((name, df))
            out.append(f"Sheet: {name}\n{df.head().to_string()}")
        return "\n".join(out), tables
    except:
        return "", []

def extract_text_from_pdf(data):
    try:
        pdf = pdfplumber.open(io.BytesIO(data))
        pages = [pg.extract_text() or "" for pg in pdf.pages]
        text = "\n".join(pages)
        if text.strip(): return text
    except: pass

    # fallback to OCR
    try:
        imgs = convert_from_bytes(data)
        return "\n".join([pytesseract.image_to_string(i) for i in imgs])
    except:
        return ""

def extract_text(file):
    name = file.name.lower()
    data = file.read()

    result = {"text": "", "images": [], "tables": []}

    if name.endswith(".pdf"):
        result["text"] = extract_text_from_pdf(data)
        try: result["images"] = convert_from_bytes(data)[:2]
        except: pass

    elif name.endswith(".docx"):
        result["text"] = extract_text_from_docx(data)

    elif name.endswith(".txt"):
        result["text"] = extract_text_from_txt(data)

    elif name.endswith(".xlsx"):
        t, tables = extract_text_from_excel(data)
        result["text"] = t
        result["tables"] = tables

    elif name.endswith((".png",".jpg",".jpeg")):
        result["text"] = extract_text_from_image(data)
        result["images"] = [Image.open(io.BytesIO(data))]

    return result


# ============================================================
# COMPARISON LOGIC
# ============================================================
def seq_diff(a, b):
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    matcher = SequenceMatcher(None, a, b)
    diff = list(unified_diff(a_lines, b_lines, lineterm=""))
    return matcher.ratio(), diff, a_lines, b_lines

def semantic_similarity(a,b):
    if not a.strip() or not b.strip(): return 0
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

def extract_numbers(text):
    return re.findall(r"[0-9.,/-]+", text)

def compare_numbers(a,b):
    na, nb = set(extract_numbers(a)), set(extract_numbers(b))
    return {"added":sorted(nb-na), "removed":sorted(na-nb), "common":sorted(na&nb)}

def compare_images(imgA, imgB):
    if not imgA or not imgB: return {"error":"No images"}
    A, B = imgA[0], imgB[0]
    return {
        "phash_distance": int(imagehash.phash(A) - imagehash.phash(B)),
        "ssim": float(ssim(np.array(ImageOps.grayscale(A).resize((256,256))),
                          np.array(ImageOps.grayscale(B).resize((256,256)))))
    }

def html_diff(a, b):
    return "<div class='card'>" + HtmlDiff().make_table(a, b) + "</div>"


# ============================================================
# NAVIGATION
# ============================================================
st.sidebar.markdown("""
<div class='sidebar-title'>
  <h2>DiffPro AI</h2>
  <p>Document Comparator</p>
</div>
""", unsafe_allow_html=True)

nav = st.sidebar.radio("", ["📄 Compare Documents", "✨ Features", "👩‍💼 About Me"])

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

    col1, col2 = st.columns(2)
    fileA = col1.file_uploader("Upload Document A", type=['pdf','docx','txt','xlsx','png','jpg'])
    fileB = col2.file_uploader("Upload Document B", type=['pdf','docx','txt','xlsx','png','jpg'])

    if fileA and fileB and st.button("Run Comparison"):

        fileA.seek(0); A = extract_text(fileA)
        fileB.seek(0); B = extract_text(fileB)

        st.success("Extraction completed.")

        # TEXT COMPARISON
        st.header("📝 Text Comparison")
        ratio, diff, a_lines, b_lines = seq_diff(A["text"], B["text"])
        sem = semantic_similarity(A["text"], B["text"])

        col1, col2 = st.columns(2)
        col1.metric("Text Similarity", f"{ratio:.3f}")
        col2.metric("Semantic Similarity", f"{sem:.3f}")

        st.subheader("Inline Visual Diff")
        st.markdown(html_diff(a_lines[:400], b_lines[:400]), unsafe_allow_html=True)

        # NUMBERS
        st.header("🔢 Numeric Comparison")
        st.json(compare_numbers(A["text"], B["text"]))

        # IMAGES
        st.header("🖼 Image Comparison")
        st.json(compare_images(A["images"], B["images"]))

        # RAW TEXTS
        colA, colB = st.columns(2)
        colA.text_area("Document A", A["text"], height=280)
        colB.text_area("Document B", B["text"], height=280)



# ============================================================
# PAGE: FEATURES
# ============================================================
elif page == "Features":

    st.title("✨ Features of DiffPro AI")

    st.markdown("""
<div class='card'>
    <h3>🔍 Intelligent Text Comparison</h3>
    <p>
        Detects exact, partial, and structural text changes using sequence matching.
        Useful for legal, academic, and revision tracking.
    </p>
</div>

<div class='card'>
    <h3>🧠 AI Semantic Understanding</h3>
    <p>
        Uses transformer embeddings to detect changes in meaning, paraphrasing,
        and rewritten content — even if the text is not identical.
    </p>
</div>

<div class='card'>
    <h3>📑 Inline Diff Viewer</h3>
    <p>
        Color-coded HTML diff shows:
        <br>🟩 Added text
        <br>🟥 Removed text
        <br>🟨 Modified text
    </p>
</div>

<div class='card'>
    <h3>🖼 OCR + Image Analysis</h3>
    <p>
        Extracts text from scanned PDFs/images and compares visuals using pHash &
        SSIM — great for reports, photos, and scanned forms.
    </p>
</div>

<div class='card'>
    <h3>📊 Excel & Table Comparison</h3>
    <p>
        Detects sheet changes, column drift, row mismatches, and cell-level anomalies.
        Very useful for finance, auditing, MIS, and analytics teams.
    </p>
</div>

st.markdown("""
<div class='card'>
    <h3>🌐 Multi-format Document Support</h3>
    <p>
        Works with PDF, DOCX, TXT, XLSX, PNG, JPG — plug and play.
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

            I design intelligent applications using ML, NLP, Vision, and UI systems.
            DiffPro AI was built to help individuals, teams, and businesses compare
            documents with clarity, accuracy, and intelligence.<br><br>

            <strong>Expertise:</strong><br>
            • Machine Learning & AI<br>
            • NLP & Document Intelligence<br>
            • OCR + Embeddings + Vision Models<br>
            • Data Engineering & Deployment<br><br>

            <strong>GitHub:</strong><br>
            <a href='https://github.com/indureddy20'>github.com/indureddy20</a>
            </p>

        </div>

    </div>
</div>
""", unsafe_allow_html=True)
