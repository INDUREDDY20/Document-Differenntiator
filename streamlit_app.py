###############################################################
# DiffPro AI — Final Fixed (Part 1 / 4)
# Imports, config, CSS, and extraction helpers
###############################################################

import streamlit as st
import io, re, json, base64, tempfile, time
from typing import List, Dict, Any

# Core libs
import nltk     # used only for compatibility if present, but we won't call punkt
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import pdfplumber
import docx
from pdf2image import convert_from_bytes
import pytesseract
import hashlib
import difflib

# Page config
st.set_page_config(
    page_title="DiffPro AI - Document Comparator",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# NOTE: We will not rely on nltk.download('punkt') because some hosts block it.
# We keep the import for environments where punkt is already present.
# For sentence splitting we use a fallback fast_sent_tokenize defined in Part 2.

# -------------------------
# GLOBAL CSS (UI preserved)
# -------------------------
st.markdown(
    """
<style>
html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg,#0A0A1F,#1A1A3A,#2D1B69);
    color:#E8E8FF;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#131324,#0C0C18);
}
.sidebar-title { text-align:center; padding:10px 5px; }
.sidebar-title h2 { color:#00D2FF; margin:0; font-weight:800; }
.sidebar-title p { color:#AAB4FF; margin-top:4px; }

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

.card {
    background: rgba(255,255,255,0.05);
    border-radius:18px;
    padding:24px;
    margin-bottom:20px;
    border:1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(12px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
}

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

.about-img {
    width:200px; height:200px; border-radius:50%;
    border:3px solid #00D2FF;
    box-shadow:0 0 25px rgba(0,210,255,0.45);
}

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

# -------------------------
# EXTRACTION UTILITIES
# -------------------------

def extract_text_from_docx(raw: bytes) -> str:
    """Extract text from DOCX bytes."""
    try:
        doc = docx.Document(io.BytesIO(raw))
        return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    except Exception:
        return ""

def extract_text_from_txt(raw: bytes) -> str:
    """Decode text file bytes."""
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return str(raw)

def extract_text_from_image(raw: bytes) -> str:
    """Run OCR on an image (PIL + pytesseract)."""
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return pytesseract.image_to_string(img)
    except Exception:
        return ""

def extract_text_from_excel(raw: bytes):
    """Return combined sheet previews and tables list."""
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
    """Try text extraction via pdfplumber, fallback to OCR images if needed."""
    try:
        pdf = pdfplumber.open(io.BytesIO(raw))
        pages = [pg.extract_text() or "" for pg in pdf.pages]
        text = "\n".join(pages)
        if text.strip():
            return text
    except Exception:
        pass

    # fallback: render pages to images and OCR
    try:
        imgs = convert_from_bytes(raw)
        texts = []
        for img in imgs:
            texts.append(pytesseract.image_to_string(img))
        return "\n".join(texts)
    except Exception:
        return ""

def extract_text(file) -> Dict[str, Any]:
    """
    Unified extractor. Returns dict with keys: text, images (list of PIL), tables (list).
    Supports: pdf, docx, txt, xlsx, png/jpg.
    """
    name = file.name.lower()
    data = file.read()

    result = {"text": "", "images": [], "tables": []}

    if name.endswith(".pdf"):
        result["text"] = extract_text_from_pdf(data)

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
        try:
            result["images"] = [Image.open(io.BytesIO(data))]
        except Exception:
            result["images"] = []

    return result

# End of Part 1
###############################################################
# Part 2 / 4 — Core Utilities (Correct Order)
###############################################################

# -------------------------------------------------------------
# 1️⃣ FAST SENTENCE TOKENIZER (NO NLTK DOWNLOAD REQUIRED)
# -------------------------------------------------------------
def fast_sent_tokenize(text: str):
    """
    A lightweight, regex-based sentence splitter.
    Replaces nltk.sent_tokenize to avoid punkt_tab errors on Streamlit Cloud.
    """
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


# -------------------------------------------------------------
# 2️⃣ CHUNK TEXT (for hash-based diff)
# -------------------------------------------------------------
def chunk_text(text: str, size=600):
    """Break large text into chunks of uniform size."""
    text = text.replace("\n\n", "\n")
    chunks = []
    for i in range(0, len(text), size):
        ck = text[i:i + size].strip()
        if ck:
            chunks.append(ck)
    return chunks


# -------------------------------------------------------------
# 3️⃣ HASH CHUNK
# -------------------------------------------------------------
def hash_chunk(chunk: str):
    """Fast MD5 hashing."""
    return hashlib.md5(chunk.encode()).hexdigest()


# -------------------------------------------------------------
# 4️⃣ HYBRID COMPARE ENGINE (FAST + ACCURATE)
# -------------------------------------------------------------
def hybrid_compare(textA: str, textB: str):
    """
    Step 1: Chunk and compare using hash (fast)
    Step 2: For chunks that changed, run smart sentence diff (accurate)
    """
    chunksA = chunk_text(textA)
    chunksB = chunk_text(textB)

    hashesA = {hash_chunk(c): c for c in chunksA}
    hashesB = {hash_chunk(c): c for c in chunksB}

    added = []
    removed = []
    potential_modified = []

    # Detect removed
    for h, c in hashesA.items():
        if h not in hashesB:
            removed.append(c)

    # Detect added
    for h, c in hashesB.items():
        if h not in hashesA:
            added.append(c)

    # Detect modified pairs
    for old in removed:
        for new in added:
            if old[:50] in new or new[:50] in old:
                potential_modified.append((old, new))

    # Smart sentence-level diff for modified chunks
    modified_final = []
    for old_chunk, new_chunk in potential_modified:
        old_sents = fast_sent_tokenize(old_chunk)
        new_sents = fast_sent_tokenize(new_chunk)

        for s1 in old_sents:
            if not new_sents:
                continue
            best = max(new_sents, key=lambda s2: difflib.SequenceMatcher(None, s1, s2).ratio())
            score = difflib.SequenceMatcher(None, s1, best).ratio()
            if score < 0.75:
                modified_final.append((s1, best))

    # Clean lists (avoid duplicates)
    added_clean = [x for x in added if all(x != m[1] for m in potential_modified)]
    removed_clean = [x for x in removed if all(x != m[0] for m in potential_modified)]

    return added_clean, removed_clean, modified_final


# -------------------------------------------------------------
# 5️⃣ CLEAN SUMMARY FORMATTER
# -------------------------------------------------------------
def clean_summary(added, removed, modified):
    out = "### 🧾 Clean Difference Summary\n\n"

    out += "## ➕ Added\n"
    out += "_No added content._\n" if not added else "\n".join([f"- {s[:300]}" for s in added])

    out += "\n\n## ➖ Removed\n"
    out += "_No removed content._\n" if not removed else "\n".join([f"- {s[:300]}" for s in removed])

    out += "\n\n## 🔁 Modified\n"
    if not modified:
        out += "_No modified content._"
    else:
        for old, new in modified:
            out += f"**Original:** {old[:300]}<br>**Changed To:** {new[:300]}<br><br>"

    return out


# -------------------------------------------------------------
# 6️⃣ SECTION / CHAPTER DIFF
# -------------------------------------------------------------
def detect_sections(text: str):
    sections = []
    for line in text.split("\n"):
        L = line.strip()
        if L.startswith(("Chapter", "CHAPTER", "#", "##")):
            sections.append(L)
        elif L.isupper() and len(L) > 10:
            sections.append(L)
    return sections


def section_diff(textA: str, textB: str):
    sa = set(detect_sections(textA))
    sb = set(detect_sections(textB))

    return list(sb - sa), list(sa - sb)


# -------------------------------------------------------------
# 7️⃣ Side-by-Side Highlight Engine
# -------------------------------------------------------------
def highlight_changes(textA, textB, added, removed, modified):

    def mark(text, items, color):
        for x in items:
            if isinstance(x, tuple):
                x = x[0]
            if x and x in text:
                text = text.replace(x, f"<span style='background:{color}'>{x}</span>")
        return text

    left = textA
    right = textB

    left = mark(left, removed, "rgba(255,0,0,0.35)")
    right = mark(right, added, "rgba(0,255,0,0.35)")

    for old, new in modified:
        if old in left:
            left = left.replace(old, f"<span style='background:rgba(255,255,0,0.35)'>{old}</span>")
        if new in right:
            right = right.replace(new, f"<span style='background:rgba(255,255,0,0.35)'>{new}</span>")

    return left, right


# -------------------------------------------------------------
# 8️⃣ EXPORT REPORT (TXT ONLY — STREAMLIT CLOUD SAFE)
# -------------------------------------------------------------
def export_report(added, removed, modified, speed_stats):
    report = "===== DiffPro AI Report =====\n\n"

    def add_section(title, content):
        nonlocal report
        report += f"\n\n### {title} ###\n"
        if not content:
            report += "No content.\n"
        else:
            for c in content:
                if isinstance(c, tuple):
                    report += f"Original: {c[0]}\nChanged To: {c[1]}\n\n"
                else:
                    report += f"- {c}\n"

    add_section("Added Content", added)
    add_section("Removed Content", removed)
    add_section("Modified Content", modified)

    report += "\n\n### Performance Stats ###\n"
    for k, v in speed_stats.items():
        report += f"{k}: {v:.4f} sec\n"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    with open(tmp.name, "w", encoding="utf-8") as f:
        f.write(report)

    return tmp.name

# End of Part 2
###############################################################
# PART 3 / 4 — COMPARE DOCUMENTS PAGE
###############################################################

# Sidebar Navigation
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


###############################################################
# PAGE: COMPARE DOCUMENTS
###############################################################
if page == "Compare Documents":

    st.title("📄 DiffPro AI — Compare Documents (Hybrid Mode)")

    debug_mode = st.checkbox("Enable Debug Mode (Show Processing Time)", value=False)

    colA, colB = st.columns(2)
    with colA:
        fileA = st.file_uploader(
            "Upload Document A",
            type=["pdf", "docx", "txt", "xlsx", "png", "jpg"]
        )
    with colB:
        fileB = st.file_uploader(
            "Upload Document B",
            type=["pdf", "docx", "txt", "xlsx", "png", "jpg"]
        )

    if fileA and fileB and st.button("Run Comparison"):

        t_start = time.time()

        # ---------------------------
        # Extract text
        # ---------------------------
        t0 = time.time()
        fileA.seek(0)
        fileB.seek(0)

        A = extract_text(fileA)
        B = extract_text(fileB)

        textA = A["text"]
        textB = B["text"]

        t_extract = time.time()

        # ---------------------------
        # Run hybrid diff
        # ---------------------------
        t1 = time.time()
        added, removed, modified = hybrid_compare(textA, textB)
        t_compare = time.time()

        # ---------------------------
        # Summary
        # ---------------------------
        summary_html = clean_summary(added, removed, modified)
        t_summary = time.time()

        # ---------------------------
        # Section-level diff
        # ---------------------------
        s_added, s_removed = section_diff(textA, textB)
        t_sections = time.time()

        # ---------------------------
        # Highlighted view
        # ---------------------------
        leftH, rightH = highlight_changes(textA, textB, added, removed, modified)
        t_highlight = time.time()

        t_end = time.time()

        speed_stats = {
            "Extraction Time": t_extract - t0,
            "Hybrid Comparison": t_compare - t1,
            "Summary Generation": t_summary - t_compare,
            "Section Diff Time": t_sections - t_summary,
            "Highlight Rendering": t_highlight - t_sections,
            "TOTAL Time": t_end - t_start
        }

        ###############################################################
        # DISPLAY RESULTS
        ###############################################################

        st.header("📝 Clean Difference Summary")
        st.markdown(summary_html, unsafe_allow_html=True)

        st.header("📚 Section / Chapter Level Differences")
        colS1, colS2 = st.columns(2)

        with colS1:
            st.subheader("➕ Added Sections")
            st.write(s_added if s_added else "No new sections found.")

        with colS2:
            st.subheader("➖ Removed Sections")
            st.write(s_removed if s_removed else "No removed sections found.")

        st.header("🖍 Side-by-Side Highlighted Comparison")
        colH1, colH2 = st.columns(2)

        with colH1:
            st.markdown(
                f"<div style='padding:12px; background:rgba(255,255,255,0.06); border-radius:10px;'>{leftH}</div>",
                unsafe_allow_html=True
            )

        with colH2:
            st.markdown(
                f"<div style='padding:12px; background:rgba(255,255,255,0.06); border-radius:10px;'>{rightH}</div>",
                unsafe_allow_html=True
            )

        st.header("📄 Raw Extracted Text (Preview)")

        colR1, colR2 = st.columns(2)
        colR1.text_area("Document A (text)", textA[:5000], height=300)
        colR2.text_area("Document B (text)", textB[:5000], height=300)

        # ---------------------------
        # Debug Output
        # ---------------------------
        if debug_mode:
            st.header("⏱ Debug Performance Stats")
            st.json(speed_stats)

        # ---------------------------
        # TXT EXPORT
        # ---------------------------
        st.header("📄 Export Results")
        if st.button("Download Report (.txt)"):
            rpt_path = export_report(added, removed, modified, speed_stats)
            with open(rpt_path, "rb") as f:
                st.download_button(
                    "Download DiffPro Report (.txt)",
                    f,
                    file_name="DiffPro_Report.txt"
                )


###############################################################
# END OF PART 3
###############################################################
###############################################################
# PART 4 / 4 — FEATURES PAGE + ABOUT PAGE + FOOTER
###############################################################

# -------------------------
# PAGE: FEATURES
# -------------------------
elif page == "Features":
    st.title("✨ Features of DiffPro AI")

    features = [
        ("Hybrid Diff Engine", 
         "Combines fast hashing + accurate sentence comparison.", "#00d2ff"),
        ("TXT Report Export", 
         "Streamlit Cloud compatible report export with clear diff text.", "#7c4dff"),
        ("Side-by-Side Highlights", 
         "Visual marking of added, removed, and modified text.", "#00bfa5"),
        ("Section / Chapter Diff", 
         "Detects large structural changes in documents.", "#ffd93d"),
        ("OCR for PDFs & Images", 
         "Extracts text from scanned PDF pages or images.", "#ff6b6b"),
        ("Excel Sheet Parsing", 
         "Reads .xlsx and displays sheet structure & preview.", "#4caf50"),
        ("Fast Processing", 
         "Optimized hybrid engine ensures high performance.", "#e91e63"),
        ("Debug Mode", 
         "Detailed timing breakdown for performance tuning.", "#ffa726"),
        ("Modern UI", 
         "Clean, stylish, gradient-powered interface.", "#29b6f6"),
    ]

    cols = st.columns(3, gap="large")

    for i, (title, desc, color) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.04);
                    padding:22px;
                    border-radius:18px;
                    border:1px solid rgba(255,255,255,0.1);
                    box-shadow:0 6px 18px rgba(0,0,0,0.4);
                    margin-bottom:20px;
                    transition:0.3s;
                ">
                    <div style="font-size:1.2rem; font-weight:800; color:white;">
                        <span style="display:inline-block; width:14px; height:14px; 
                            background:{color}; border-radius:4px; margin-right:10px;">
                        </span>
                        {title}
                    </div>
                    <div style="color:#D6D6FF; margin-top:6px; font-size:0.95rem;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# -------------------------
# PAGE: ABOUT ME
# -------------------------
elif page == "About Me":

    st.title("👩‍💼 About the Creator")

    col1, col2 = st.columns([1,2], gap="large")

    with col1:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/2922/2922561.png",
            width=200,
            caption="Indu Reddy"
        )

    with col2:
        st.markdown(
            """
            <div style="background: rgba(255,255,255,0.05);
                padding:20px; border-radius:16px;
                border:1px solid rgba(255,255,255,0.08);">

            <h2 style="color:#00D2FF; margin:0;">Indu Reddy</h2>
            <p style="color:#EAEAFF; font-size:1.05rem; line-height:1.7;">

            <strong>AI Engineer • NLP Developer • Python Specialist</strong><br><br>

            I build tools that make document understanding easier, faster, and more accurate.  
            DiffPro AI combines NLP, OCR, hybrid diff algorithms, and stylish UI to create a top-tier comparison engine.

            <br><br><strong>Skills:</strong><br>
            • Machine Learning & AI<br>
            • Natural Language Processing<br>
            • OCR & Pattern Recognition<br>
            • Full-Stack Python Development<br>
            • UI/UX Engineering<br><br>

            <strong>GitHub:</strong> 
            <a href="https://github.com/indureddy20" target="_blank" style="color:#7C4DFF;">
            github.com/indureddy20</a>

            </p></div>
            """,
            unsafe_allow_html=True,
        )


# -------------------------
# FOOTER
# -------------------------
st.markdown(
    """
<div class='footer'>
    Built with ❤️ and passion — DiffPro AI Hybrid Engine v2
</div>
""",
    unsafe_allow_html=True,
)
