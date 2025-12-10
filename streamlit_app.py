###############################################################
# DiffPro AI – Final Hybrid Version 
# UI Preserved + Ultra Fast Hybrid Diff + PDF Export + Highlights
###############################################################

import streamlit as st
import io, re, json, base64, tempfile, time
from typing import List, Dict, Any

import nltk
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import pdfplumber
import docx
from pdf2image import convert_from_bytes
import pytesseract
import hashlib
import difflib



###############################################################
# PAGE CONFIG
###############################################################
st.set_page_config(
    page_title="DiffPro AI - Document Comparator",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

nltk.download("punkt", quiet=True)

###############################################################
# GLOBAL CSS (Original UI preserved)
###############################################################
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

###############################################################
# EXTRACTION FUNCTIONS
###############################################################

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

    return result


###############################################################
# HYBRID COMPARISON ENGINE
###############################################################

def hybrid_compare(textA, textB):
    chunksA = chunk_text(textA)
    chunksB = chunk_text(textB)

    hashesA = {hash_chunk(c): c for c in chunksA}
    hashesB = {hash_chunk(c): c for c in chunksB}

    added, removed, potential_modified = [], [], []

    for h, c in hashesA.items():
        if h not in hashesB:
            removed.append(c)

    for h, c in hashesB.items():
        if h not in hashesA:
            added.append(c)

    for r in removed:
        for a in added:
            if r[:60] in a or a[:60] in r:
                potential_modified.append((r, a))

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

    added_clean = [x for x in added if all(x != m[1] for m in potential_modified)]
    removed_clean = [x for x in removed if all(x != m[0] for m in potential_modified)]

    return added_clean, removed_clean, modified_final


###############################################################
# CLEAN SUMMARY FORMATTER
###############################################################
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


###############################################################
# SECTION / CHAPTER LEVEL DIFFERENCE
###############################################################
def detect_sections(text):
    sections = []
    for line in text.split("\n"):
        L = line.strip()
        if L.startswith(("Chapter", "CHAPTER", "#", "##")):
            sections.append(L)
        elif L.isupper() and len(L) > 10:
            sections.append(L)
    return sections


def section_diff(textA, textB):
    sa = set(detect_sections(textA))
    sb = set(detect_sections(textB))
    return list(sb - sa), list(sa - sb)


###############################################################
# SIDE-BY-SIDE HIGHLIGHT ENGINE
###############################################################
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

    # Removed from A → highlight red
    left = mark(left, removed, "rgba(255,0,0,0.35)")

    # Added in B → highlight green
    right = mark(right, added, "rgba(0,255,0,0.35)")

    # Modified → highlight yellow
    for old, new in modified:
        if old in left:
            left = left.replace(old, f"<span style='background:rgba(255,255,0,0.35)'>{old}</span>")
        if new in right:
            right = right.replace(new, f"<span style='background:rgba(255,255,0,0.35)'>{new}</span>")

    return left, right

###############################################################
# EXPORT REPORT AS TXT (Streamlit Cloud Compatible)
###############################################################
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
# PAGE: COMPARE DOCUMENTS (HYBRID DIFF VERSION)
###############################################################
if page == "Compare Documents":

    st.title("📄 DiffPro AI — Compare Any Two Documents (Hybrid Mode + Highlights + PDF)")

    # Debug mode toggle
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
        
        # ==== TIMER START ====
        t_start = time.time()

        # Extract Text
        t0 = time.time()
        fileA.seek(0)
        fileB.seek(0)
        A = extract_text(fileA)
        B = extract_text(fileB)
        t_extract = time.time()

        st.success("Extraction complete.")

        textA = A["text"]
        textB = B["text"]

        # ==== Hybrid Diff ====
        t1 = time.time()
        added, removed, modified = hybrid_compare(textA, textB)
        t_compare = time.time()

        # ==== Summary Build ====
        summary_html = clean_summary(added, removed, modified)
        t_summary = time.time()

        # ==== Section / Chapter Diff ====
        s_added, s_removed = section_diff(textA, textB)
        t_sections = time.time()

        # ==== Side-by-Side Highlights ====
        leftH, rightH = highlight_changes(textA, textB, added, removed, modified)
        t_highlight = time.time()

        # ==== END TIMER ====
        t_end = time.time()

        speed_stats = {
            "Extraction Time": t_extract - t0,
            "Hybrid Comparison": t_compare - t1,
            "Summary Generation": t_summary - t_compare,
            "Section Diff Time": t_sections - t_summary,
            "Highlight Rendering": t_highlight - t_sections,
            "Total Processing Time": t_end - t_start,
        }

        ###############################################################
        # SHOW CLEAN SUMMARY
        ###############################################################
        st.header("📝 Clean Text Differences")
        st.markdown(summary_html, unsafe_allow_html=True)

        ###############################################################
        # SECTION / CHAPTER DIFFERENCES
        ###############################################################
        st.header("📚 Section / Chapter Level Changes")

        colS1, colS2 = st.columns(2)
        with colS1:
            st.subheader("➕ Added Sections")
            st.write(s_added if s_added else "No new sections found.")

        with colS2:
            st.subheader("➖ Removed Sections")
            st.write(s_removed if s_removed else "No removed sections found.")

        ###############################################################
        # SIDE-BY-SIDE HIGHLIGHT PANEL
        ###############################################################
        st.header("🖍 Side-by-Side Highlighted View")

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

        ###############################################################
        # RAW TEXT PANELS
        ###############################################################
        st.header("📄 Raw Extracted Text (Preview)")

        colR1, colR2 = st.columns(2)
        colR1.text_area("Document A (extracted)", textA[:5000], height=300)
        colR2.text_area("Document B (extracted)", textB[:5000], height=300)

        ###############################################################
        # DEBUG MODE OUTPUT
        ###############################################################
        if debug_mode:
            st.header("⏱ Performance Debug Stats")
            st.json(speed_stats)

        ###############################################################
        # TXT REPORT EXPORT BUTTON
        ###############################################################
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
# PAGE: FEATURES
###############################################################
elif page == "Features":
    st.title("✨ Features of DiffPro AI")

    features = [
        ("Hybrid Diff Engine", 
         "Combines ultra-fast hashing with smart sentence-level comparison.", "#00d2ff"),

        ("PDF Report Export", 
         "Generate clean, formatted diff reports for documentation and auditing.", "#7c4dff"),

        ("Side-by-Side Highlights", 
         "Instant visual detection of added, removed, and modified content.", "#00bfa5"),

        ("Section / Chapter-Level Diff", 
         "Detects changed headings, chapters, and major document structure edits.", "#ffd93d"),

        ("OCR for Scanned PDFs & Images", 
         "Extracts text from scanned images using advanced OCR techniques.", "#ff6b6b"),

        ("Excel Table Support", 
         "Extracts sheets and table previews directly from .xlsx files.", "#4caf50"),

        ("Modern UI Experience", 
         "Sleek gradient UI, glassmorphism panels, and responsive layout.", "#e91e63"),

        ("Debug Performance Mode", 
         "Shows complete internal timings for optimization & debugging.", "#ffa726"),

        ("100% Streamlit Cloud Compatible", 
         "No external dependencies. Fast and lightweight.", "#29b6f6"),
    ]

    cols = st.columns(3, gap="large")

    for i, (title, desc, accent) in enumerate(features):
        col = cols[i % 3]
        with col:
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.03);
                    padding:26px;
                    border-radius:18px;
                    min-height:160px;
                    border:1px solid rgba(255,255,255,0.08);
                    box-shadow:0 10px 25px rgba(0,0,0,0.5);
                    transition:0.3s;
                ">
                    <div style="font-size:1.2rem; font-weight:800; color:white;">
                        <span style="display:inline-block; width:12px; height:12px; background:{accent};
                        border-radius:3px; margin-right:10px;"></span>
                        {title}
                    </div>
                    <div style="color:rgba(220,220,255,0.9); margin-top:8px; font-size:0.95rem;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


###############################################################
# PAGE: ABOUT ME
###############################################################
elif page == "About Me":

    st.title("👩‍💼 About the Creator")

    col_img, col_text = st.columns([1, 2], gap="large")

    with col_img:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/2922/2922561.png",
            width=240,
            caption="Indu Reddy"
        )

    with col_text:
        st.markdown(
            """
            <div style="background: rgba(255,255,255,0.04);
                padding:20px; border-radius:14px; 
                border:1px solid rgba(255,255,255,0.05);">

            <h2 style="color:#00D2FF; margin:0;">Indu Reddy</h2>

            <p style="color:#E8E8FF; font-size:1.1rem; line-height:1.7;">
            <strong>AI Engineer • Document Intelligence • Python Developer</strong><br><br>

            I build intelligent tools that enhance productivity using NLP, OCR, 
            fast text diffing, hybrid analysis models, and user-centric UI design.<br><br>

            DiffPro AI is engineered to solve the hardest document comparison 
            problems with unmatched speed and clarity.<br><br>

            <strong>Expertise:</strong><br>
            • AI & Machine Learning<br>
            • NLP & Text Embedding<br>
            • OCR & Image Processing<br>
            • Full-Stack Python Development<br>
            • Intelligent UI/UX Engineering<br><br>

            <strong>GitHub:</strong> 
            <a href="https://github.com/indureddy20" style="color:#7C4DFF;">github.com/indureddy20</a>
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )


###############################################################
# FOOTER
###############################################################
st.markdown(
    """
<div class='footer'>
    Built with ❤️ and passion — DiffPro AI Hybrid Engine v2
</div>
""",
    unsafe_allow_html=True,
)


