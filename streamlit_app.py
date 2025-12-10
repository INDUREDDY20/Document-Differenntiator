# ============================
# DiffPro AI – Clean Summary Version 
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

from rapidfuzz import fuzz

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
# GLOBAL CSS – KEEP ORIGINAL UI
# ============================
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

.about-container { display:flex; align-items:center; gap:2rem; }
.about-img {
    width:200px; height:200px; border-radius:50%;
    border:3px solid #00D2FF;
    box-shadow:0 0 25px rgba(0,210,255,0.45);
}
.about-text { font-size:1.2rem; line-height:1.7; color:#E0E0FF; }

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
# CLEAN DIFFERENCE COMPARISON
# ============================
def split_sentences(text):
    try:
        return nltk.sent_tokenize(text)
    except:
        return text.split(".")


def compare_sentences(old_sent, new_sent, threshold=80):
    added, removed, modified = [], [], []

    for s1 in old_sent:
        best = max(new_sent, key=lambda s2: fuzz.ratio(s1, s2)) if new_sent else ""
        score = fuzz.ratio(s1, best)

        if score < threshold:
            if score < 30:
                removed.append(s1)
            else:
                modified.append((s1, best))

    for s2 in new_sent:
        best = max(old_sent, key=lambda s1: fuzz.ratio(s2, s1)) if old_sent else ""
        score = fuzz.ratio(s2, best)

        if score < threshold:
            if score < 30:
                added.append(s2)

    return added, removed, modified


def clean_summary(added, removed, modified):
    out = "### 🧾 Clean Difference Summary\n\n"

    out += "## ➕ Added\n"
    out += "_No added content found._\n" if not added else "\n".join([f"- {s}" for s in added])

    out += "\n\n## ➖ Removed\n"
    out += "_No removed content found._\n" if not removed else "\n".join([f"- {s}" for s in removed])

    out += "\n\n## 🔁 Modified\n"
    if not modified:
        out += "_No modified content found._"
    else:
        for old, new in modified:
            out += f"**Original:** {old}\n\n**Changed To:** {new}\n\n"

    return out


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
# PAGE: COMPARE DOCUMENTS (UPDATED)
# ============================
if page == "Compare Documents":
    st.title("📄 DiffPro AI — Compare Any Two Documents (Clean Summary Mode)")

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

        textA = A["text"]
        textB = B["text"]

        sentA = split_sentences(textA)
        sentB = split_sentences(textB)

        added, removed, modified = compare_sentences(sentA, sentB)

        st.header("📝 Clean Text Differences")
        st.markdown(clean_summary(added, removed, modified))

        col1, col2 = st.columns(2)
        col1.text_area("Document A — Extracted", textA[:5000], height=300)
        col2.text_area("Document B — Extracted", textB[:5000], height=300)


# ============================
# PAGE: FEATURES
# ============================
elif page == "Features":
    st.title("✨ Features of DiffPro AI")

    features = [
        ("AI Clean Comparison", "Shows Added, Removed, Modified sentences clearly.", "#00d2ff"),
        ("OCR for Scanned PDFs", "Extract text from scanned PDFs/images using Tesseract.", "#7c4dff"),
        ("Excel & Table Support", "Preview sheet names and tables.", "#00bfa5"),
        ("Modern UI Experience", "Polished glassmorphism UI.", "#ffd93d"),
        ("Fast Processing", "Optimized sentence-level comparison.", "#4caf50"),
    ]

    cols = st.columns(3, gap="large")
    for i, (title, desc, accent) in enumerate(features):
        col = cols[i % 3]
        with col:
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.02);
                    border-radius:16px; padding:28px; min-height:160px;
                    box-shadow:0 12px 36px rgba(2,8,30,0.6);
                    border:1px solid rgba(255,255,255,0.04);
                ">
                    <div style="font-weight:800; font-size:1.2rem; color:#ffffff; margin-bottom:10px;">
                        <span style="display:inline-block; width:12px; height:12px; background:{accent}; border-radius:3px; margin-right:10px;"></span>
                        {title}
                    </div>
                    <div style="color: rgba(220,220,255,0.9); font-size:0.95rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================
# PAGE: ABOUT ME
# ============================
elif page == "About Me":
    st.title("👩‍💼 About the Creator")

    col_img, col_text = st.columns([1, 2], gap="large")

    with col_img:
        avatar_url = "https://cdn-icons-png.flaticon.com/512/2922/2922561.png"
        st.image(avatar_url, width=240, caption="Indu Reddy", use_column_width=False)

    with col_text:
        st.markdown(
            """
            <div style="background: rgba(255,255,255,0.02); padding:18px; border-radius:12px;
                border:1px solid rgba(255,255,255,0.03);">
              <h2 style="color:#00D2FF; margin-top:0; margin-bottom:6px;">Indu Reddy</h2>
              <p style="color:#E8E8FF; font-size:1.05rem; line-height:1.6; margin-top:0;">
                <strong>AI Engineer • Bengaluru</strong><br><br>
                I design AI-powered tools using NLP, OCR, and Document Intelligence.<br><br>
                <strong>Expertise:</strong><br>
                • NLP & Deep Learning<br>
                • OCR & Document AI<br>
                • Python & Automation<br><br>
                <strong>GitHub:</strong>
                <a href="https://github.com/indureddy20" style="color:#7C4DFF;">github.com/indureddy20</a>
              </p>
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
