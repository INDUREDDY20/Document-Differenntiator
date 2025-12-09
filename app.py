
import streamlit as st
import streamlit.components.v1 as components
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

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="DiffPro AI - Document Comparator",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== NLTK & MODEL =====
nltk.download("punkt", quiet=True)


@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = get_model()

# ===== GLOBAL CSS (polished) =====
st.markdown(
    """
<style>
/* base */
html, body, .stApp {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    background: linear-gradient(135deg, #0b0b20 5%, #1b1840 40%, #2f1b61 100%);
    color: #eaeaff;
    min-height:100vh;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #0b0b14 100%);
    padding-top: 18px;
}
.sidebar-title { text-align:left; padding-left:16px; }
.sidebar-title h2 { color:#00d2ff; margin:0; font-weight:800; font-size:1.6rem;}
.sidebar-sub { color:#aeb8ff; padding-left:16px; margin-bottom:8px; }

/* nav radio style fallback (appearance differs across platforms) */
.stRadio > div { gap: 6px; }

/* buttons */
.stButton > button {
    background: linear-gradient(90deg,#00d2ff,#7c4dff);
    color: #fff;
    padding: 12px 32px;
    border-radius: 999px;
    border: none;
    font-weight:700;
    box-shadow: 0 8px 28px rgba(124,77,255,0.12);
}
.stButton > button:hover { transform: translateY(-3px); }

/* card */
.card {
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.04);
    box-shadow: 0 12px 40px rgba(2,6,23,0.55);
    margin-bottom: 18px;
}

/* header hero */
.app-title {
    display:flex; align-items:center; gap:14px; margin-bottom:18px;
}
.app-title h1 { margin:0; font-size:2.6rem; letter-spacing:-1px; color: #fff; font-weight:900; }
.app-sub { color:#c8cef9; margin-top:6px; }

/* diff table */
table.diff { width:100%; border-collapse:collapse; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace; }
table.diff th, table.diff td { padding:8px; border:1px solid rgba(255,255,255,0.03); vertical-align:top; }
.diff_add { background: rgba(76,175,80,0.22) !important; }
.diff_sub { background: rgba(244,67,54,0.18) !important; }
.diff_chg { background: rgba(255,202,40,0.16) !important; }

/* about */
.about-container { display:flex; gap:2rem; align-items:center; }
.about-avatar {
    width:220px; height:220px; border-radius:18px; object-fit:cover;
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
    border:4px solid rgba(255,255,255,0.03);
}
.about-text { max-width:900px; font-size:1.05rem; color:#e9e9ff; line-height:1.7; }

/* footer */
.footer {
    text-align:center; padding:18px 10px; opacity:0.9; color:#dcdcff; margin-top:36px;
}

/* responsive tweaks */
@media (max-width:900px) {
    .about-container { flex-direction:column; align-items:center; }
    .about-avatar { width:160px; height:160px; }
    .app-title h1 { font-size:2rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ===== EXTRACTION HELPERS =====
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


# ===== COMPARISON HELPERS =====
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


# ===== SIDEBAR & NAV =====
st.sidebar.markdown(
    """
<div class='sidebar-title'>
  <h2>DiffPro AI</h2>
  <p class='sidebar-sub'>Document Comparator</p>
</div>
""",
    unsafe_allow_html=True,
)

nav = st.sidebar.radio("", ["📄 Compare Documents", "✨ Features", "👩‍💼 About Me"])
page = "Compare Documents" if nav.startswith("📄") else "Features" if nav.startswith("✨") else "About Me"


# ===== PAGE: COMPARE DOCUMENTS =====
if page == "Compare Documents":
    # Hero
    st.markdown(
        """
<div class="app-title">
  <div style="width:4rem; height:4rem; border-radius:12px; background:linear-gradient(90deg,#00d2ff,#7c4dff); display:flex; align-items:center; justify-content:center; box-shadow:0 10px 30px rgba(0,0,0,0.4);">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg"><path d="M3 3h18v2H3zM3 7h18v2H3zM3 11h18v2H3zM3 15h18v2H3z"/></svg>
  </div>
  <div>
    <h1 style="margin:0; color:white;">DiffPro AI</h1>
    <div class="app-sub">Upload any two documents — visual & semantic comparison</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Uploaders
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

        # text metrics
        st.header("📝 Text Comparison")
        ratio, diff_lines, a_lines, b_lines = seq_diff(A.get("text", ""), B.get("text", ""))
        sem_sim = semantic_similarity(A.get("text", ""), B.get("text", ""))

        m1, m2 = st.columns(2)
        m1.metric("Exact Text Similarity", f"{ratio:.3f}")
        m2.metric("Semantic Similarity", f"{sem_sim:.3f}")

        st.subheader("Visual Inline Diff")
        # show HTML diff (safe small snippet)
        diff_html = html_diff(a_lines[:400], b_lines[:400])
        st.markdown(diff_html, unsafe_allow_html=True)

        # numeric diffs
        st.header("🔢 Numeric Differences")
        st.json(compare_numbers(A.get("text", ""), B.get("text", "")))

        # images
        st.header("🖼 Image Comparison")
        img_report = compare_images(A.get("images", []), B.get("images", []))
        st.write(img_report)
        if A.get("images"):
            st.image(A["images"][0], caption="Doc A — preview", width=260)
        if B.get("images"):
            st.image(B["images"][0], caption="Doc B — preview", width=260)

        # side-by-side text
        st.header("📚 Side-by-Side Text")
        left, right = st.columns(2)
        left.text_area("Document A Text", A.get("text", "")[:100000], height=300)
        right.text_area("Document B Text", B.get("text", "")[:100000], height=300)

        # download report
        st.header("📥 Download JSON Report")
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
        href = f'<a href="data:application/json;base64,{b64}" download="diffpro_report.json" style="color:#fff; font-weight:700;">Download JSON report</a>'
        st.markdown(href, unsafe_allow_html=True)


# ===== PAGE: FEATURES (polished cards rendered via components.html to avoid any escaping) =====
elif page == "Features":
    st.markdown(
        """
<div class="app-title" style="margin-top:6px;">
  <h1 style="margin:0; font-size:2.4rem; color:white;">✨ Features of DiffPro AI</h1>
</div>
""",
        unsafe_allow_html=True,
    )

    features_html = """
<div style="display:flex; flex-direction:column; gap:18px; padding:8px 0 18px 0;">
  <div class='card'>
    <h3 style="margin-top:0;">🔍 Intelligent Text Comparison</h3>
    <p>Detects exact, partial, and structural text changes (line/paragraph level) using robust sequence matching.</p>
  </div>

  <div class='card'>
    <h3>🧠 AI Semantic Understanding</h3>
    <p>Uses transformer embeddings to detect changes in meaning, paraphrasing and rewritten content — great for paraphrase detection and semantic drift.</p>
  </div>

  <div class='card'>
    <h3>📑 Inline Diff Viewer</h3>
    <p>Color-coded inline diff with added/removed/modified highlights for fast visual inspection.</p>
  </div>

  <div class='card'>
    <h3>🖼 OCR + Image Analysis</h3>
    <p>Extracts text from scanned PDFs/images and compares visuals using perceptual hashing (pHash) and SSIM.</p>
  </div>

  <div class='card'>
    <h3>📊 Excel & Table Comparison</h3>
    <p>Sheet-level and cell-level comparison, useful for finance, audit and MIS workflows.</p>
  </div>

  <div class='card'>
    <h3>📤 JSON Comparison Report</h3>
    <p>Export structured JSON with full comparison results for automation and record-keeping.</p>
  </div>

  <div class='card'>
    <h3>🌐 Multi-format Support</h3>
    <p>Supports PDF, DOCX, TXT, XLSX, PNG, JPG — automatic handling of each format.</p>
  </div>
</div>
"""
    # Render with components to be robust (keeps your styling and avoids escaping issues)
    components.html(features_html, height=720, scrolling=True)


# ===== PAGE: ABOUT ME (premium realistic avatar) =====
elif page == "About Me":
    st.markdown(
        """
<div class="app-title" style="margin-top:6px;">
  <h1 style="margin:0; font-size:2.4rem; color:white;">👩‍💼 About the Creator</h1>
</div>
""",
        unsafe_allow_html=True,
    )

    # realistic avatar placeholder — swap this URL for a specific portrait image if desired
    realistic_avatar_url = "https://randomuser.me/api/portraits/women/33.jpg"

    about_html = f"""
<div style="padding:12px;">
  <div class='card' style="display:flex; align-items:center; gap:28px; flex-wrap:wrap;">
      <img src="{realistic_avatar_url}" class="about-avatar" />
      <div class="about-text">
          <h2 style="margin-top:0; color:#dff6ff;">Indu Reddy</h2>
          <p><strong>AI Engineer • Bengaluru</strong></p>
          <p>
            I design and deploy advanced AI systems combining <strong>Machine Learning</strong>,
            <strong>NLP</strong>, <strong>Computer Vision</strong>, and thoughtfully crafted UI.
            DiffPro AI is built to compare documents precisely — both visually and semantically.
          </p>
          <p>
            <strong>Expertise:</strong><br/>
            • Artificial Intelligence & Machine Learning<br/>
            • NLP & Document Intelligence<br/>
            • OCR, Embeddings & Vision Models<br/>
            • Data Engineering & Production Deployment
          </p>
          <p><strong>GitHub:</strong> <a href="https://github.com/indureddy20" style="color:#9ee0ff;">github.com/indureddy20</a></p>
      </div>
  </div>
</div>
"""
    components.html(about_html, height=480, scrolling=True)


# ===== GLOBAL FOOTER =====
st.markdown(
    """
<div class='footer'>
  Built with ❤️ and passion
</div>
""",
    unsafe_allow_html=True,
)
