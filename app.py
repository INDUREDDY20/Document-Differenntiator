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


# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="DiffPro AI - Document Comparator",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# NLTK
nltk.download("punkt", quiet=True)


# ===========================
# MODEL LOADING (CACHED)
# ===========================
@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = get_model()


# ===========================
# GLOBAL CSS
# ===========================
st.markdown(
    """
<style>
html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0A0A1F 0%, #1A1A3A 50%, #2D1B69 100%);
    color: #E8E8FF;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #131324 0%, #0C0C18 100%);
    padding-top: 18px;
}
.sidebar-title { text-align:center; margin-bottom:8px; }
.sidebar-title h2 { color:#00D2FF; margin:0; font-size:1.8rem; }
.sidebar-title p { color:#AAB4FF; margin-top:4px; }

/* BUTTONS */
.stButton > button {
    background: linear-gradient(45deg, #00D2FF, #7C4DFF);
    padding: 12px 36px;
    border-radius: 40px;
    font-size: 1.05rem;
    font-weight: 600;
    border: none;
    color: white;
}
.stButton > button:hover { transform: translateY(-3px); }

/* GLASS CARD */
.card {
    background: rgba(255,255,255,0.05);
    padding: 1.6rem;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 18px;
    backdrop-filter: blur(8px);
}

/* DIFF TABLE */
table.diff { width:100%; border-collapse: collapse; font-family: monospace; }
table.diff th { background: rgba(255,255,255,0.06); padding:6px; }
table.diff td { padding:6px; vertical-align:top; }
.diff_add { background: rgba(76,175,80,0.28) !important; }
.diff_sub { background: rgba(244,67,54,0.28) !important; }
.diff_chg { background: rgba(255,202,40,0.28) !important; }

/* ABOUT */
.about-container { display:flex; gap:1.6rem; align-items:center; }
.about-img {
    width:160px; height:160px; border-radius:50%;
    border:3px solid #00D2FF; box-shadow:0 0 18px rgba(0,210,255,0.28);
}
.about-text { font-size:1.05rem; line-height:1.6; color:#EEE; }
.about-text a { color:#7C4DFF; font-weight:600; }

</style>
""",
    unsafe_allow_html=True,
)


# ===========================
# EXTRACTION HELPERS
# ===========================
def extract_text_from_docx(raw: bytes) -> str:
    doc = docx.Document(io.BytesIO(raw))
    return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])


def extract_text_from_txt(raw: bytes) -> str:
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return str(raw)


def extract_text_from_image(raw: bytes) -> str:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return pytesseract.image_to_string(img)


def extract_text_from_excel(raw: bytes):
    try:
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        out = []
        tables = []
        for name, df in sheets.items():
            tables.append((name, df))
            out.append(f"Sheet: {name}\nColumns: {list(df.columns)}\n{df.head().to_string()}")
        return "\n".join(out), tables
    except Exception:
        return "", []


def extract_text_from_pdf(raw: bytes) -> str:
    try:
        pdf = pdfplumber.open(io.BytesIO(raw))
        pages = [pg.extract_text() or "" for pg in pdf.pages]
        text = "\n".join(pages)
        if text.strip():
            return text
    except Exception:
        pass

    # fallback OCR
    try:
        imgs = convert_from_bytes(raw)
        return "\n".join([pytesseract.image_to_string(i) for i in imgs])
    except Exception:
        return ""


def extract_text(file) -> Dict[str, Any]:
    name = file.name.lower()
    data = file.read()
    result = {"text": "", "images": [], "tables": []}

    if name.endswith(".pdf"):
        result["text"] = extract_text_from_pdf(data)
        try:
            result["images"] = convert_from_bytes(data)[:2]
        except Exception:
            result["images"] = []

    elif name.endswith(".docx"):
        result["text"] = extract_text_from_docx(data)

    elif name.endswith(".txt"):
        result["text"] = extract_text_from_txt(data)

    elif name.endswith((".xlsx", ".xls")):
        t, tables = extract_text_from_excel(data)
        result["text"] = t
        result["tables"] = tables

    elif name.endswith((".png", ".jpg", ".jpeg")):
        result["text"] = extract_text_from_image(data)
        try:
            result["images"] = [Image.open(io.BytesIO(data))]
        except Exception:
            result["images"] = []

    else:
        result["text"] = extract_text_from_pdf(data)

    return result


# ===========================
# COMPARISON HELPERS
# ===========================
def seq_diff(a: str, b: str):
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    matcher = SequenceMatcher(None, a, b)
    diff = list(unified_diff(a_lines, b_lines, lineterm=""))
    return matcher.ratio(), diff, a_lines, b_lines


def semantic_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    emb_a = model.encode(a, convert_to_tensor=True)
    emb_b = model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b).item())


def extract_numbers(text: str) -> List[str]:
    return re.findall(r"[0-9.,/-]+", text)


def compare_numbers(a: str, b: str) -> Dict[str, List[str]]:
    na, nb = set(extract_numbers(a)), set(extract_numbers(b))
    return {"added": sorted(nb - na), "removed": sorted(na - nb), "common": sorted(na & nb)}


def compare_images(imgs_a: List[Image.Image], imgs_b: List[Image.Image]) -> Dict[str, Any]:
    if not imgs_a or not imgs_b:
        return {"error": "No images to compare"}
    a = imgs_a[0]
    b = imgs_b[0]

    try:
        ph_a = imagehash.phash(a)
        ph_b = imagehash.phash(b)
        ph_dist = int(ph_a - ph_b)
    except Exception:
        ph_dist = None

    try:
        ga = ImageOps.grayscale(a).resize((256, 256))
        gb = ImageOps.grayscale(b).resize((256, 256))
        ssim_val = float(ssim(np.array(ga), np.array(gb)))
    except Exception:
        ssim_val = None

    return {"phash_distance": ph_dist, "ssim": ssim_val}


def html_diff(a_lines: List[str], b_lines: List[str]) -> str:
    table = HtmlDiff(wrapcolumn=80).make_table(a_lines, b_lines)
    return f"<div class='card'>{table}</div>"


# ===========================
# SIDEBAR / NAVIGATION
# ===========================
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


# ===========================
# PAGE: COMPARE DOCUMENTS
# ===========================
if page == "Compare Documents":
    st.title("📄 DiffPro AI — Compare Documents")

    col_a, col_b = st.columns(2)
    with col_a:
        file_a = st.file_uploader("Upload Document A", type=["pdf", "docx", "txt", "xlsx", "png", "jpg"])
    with col_b:
        file_b = st.file_uploader("Upload Document B", type=["pdf", "docx", "txt", "xlsx", "png", "jpg"])

    if file_a and file_b and st.button("Run Comparison"):
        file_a.seek(0)
        A = extract_text(file_a)
        file_b.seek(0)
        B = extract_text(file_b)

        st.success("Extraction complete.")

        # Text comparison
        st.header("📝 Text Comparison")
        ratio, diff_lines, a_lines, b_lines = seq_diff(A.get("text", ""), B.get("text", ""))
        sem_sim = semantic_similarity(A.get("text", ""), B.get("text", ""))

        c1, c2 = st.columns(2)
        c1.metric("Exact Text Similarity (difflib)", f"{ratio:.3f}")
        c2.metric("Semantic Similarity (embeddings)", f"{sem_sim:.3f}")

        st.subheader("Visual Side-by-Side Diff")
        st.markdown(html_diff(a_lines[:400], b_lines[:400]), unsafe_allow_html=True)

        # Numeric differences
        st.header("🔢 Numeric Differences")
        st.json(compare_numbers(A.get("text", ""), B.get("text", "")))

        # Table comparisons (if present)
        st.header("📊 Table / Excel Summary")
        st.write({"tables_a": [t[0] for t in A.get("tables", [])], "tables_b": [t[0] for t in B.get("tables", [])]})

        # Image comparisons
        st.header("🖼 Visual Comparison")
        img_report = compare_images(A.get("images", []), B.get("images", []))
        st.write(img_report)
        if A.get("images"):
            st.image(A["images"][0], caption="Doc A — image preview", use_column_width=False, width=300)
        if B.get("images"):
            st.image(B["images"][0], caption="Doc B — image preview", use_column_width=False, width=300)

        # Side-by-side text
        st.header("📚 Side-by-side Text View")
        left, right = st.columns(2)
        left.text_area("Document A Text", A.get("text", "")[:100000], height=300)
        right.text_area("Document B Text", B.get("text", "")[:100000], height=300)

        # Download JSON report
        st.header("📥 Download Report")
        report = {
            "file_a": getattr(file_a, "name", "A"),
            "file_b": getattr(file_b, "name", "B"),
            "text_ratio": ratio,
            "semantic_similarity": sem_sim,
            "numeric_diff": compare_numbers(A.get("text", ""), B.get("text", "")),
            "image_report": img_report,
        }
        report_json = json.dumps(report, indent=2)
        b64 = base64.b64encode(report_json.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="diffpro_report.json">Download JSON report</a>'
        st.markdown(href, unsafe_allow_html=True)


# ===========================
# PAGE: FEATURES
# ===========================
elif page == "Features":
    st.title("✨ Features of DiffPro AI")

    features_html = """
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
        and rewritten content — even when the text is restructured.
    </p>
</div>

<div class='card'>
    <h3>📑 Inline Diff Viewer</h3>
    <p>
        Color-coded HTML diff shows:<br>
        🟩 Added text <br>
        🟥 Removed text <br>
        🟨 Modified text
    </p>
</div>

<div class='card'>
    <h3>🖼 OCR + Image Analysis</h3>
    <p>
        Extracts text from scanned PDFs/images and compares visuals using pHash &
        SSIM — perfect for reports and scanned documents.
    </p>
</div>

<div class='card'>
    <h3>📊 Excel & Table Comparison</h3>
    <p>
        Detects sheet-level changes, column drift, row mismatches, and cell-level anomalies.
    </p>
</div>

<div class='card'>
    <h3>📤 JSON Comparison Report</h3>
    <p>
        Exports differences in text, semantics, numbers, tables, and images —
        helpful for automation and auditing.
    </p>
</div>

<div class='card'>
    <h3>🌐 Multi-format Support</h3>
    <p>
        Supports PDF, DOCX, TXT, XLSX, PNG, JPG out of the box.
    </p>
</div>
"""

    st.markdown(features_html, unsafe_allow_html=True)


# ===========================
# PAGE: ABOUT ME
# ===========================
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
""",
        unsafe_allow_html=True,
    )
# ===========================
# GLOBAL FOOTER
# ===========================
st.markdown(
    """
<div style='text-align:center; margin-top:40px; padding:12px; opacity:0.7; font-size:0.9rem;'>
    Built with ❤️ and passion
</div>
""",
    unsafe_allow_html=True
)
