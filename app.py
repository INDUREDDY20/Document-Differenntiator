import streamlit as st

# MUST BE FIRST
st.set_page_config(
    page_title="DiffPro AI - Document Comparator",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- imports ----
import io
import re
import json
import math
import tempfile
import base64
from pathlib import Path
from typing import Tuple, Dict, Any, List

import nltk
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import pdfplumber
import docx
from pdf2image import convert_from_bytes
import pytesseract
from difflib import SequenceMatcher, ndiff, unified_diff
from sentence_transformers import SentenceTransformer, util
import imagehash
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# ---- nltk downloads ----
nltk.download('punkt', quiet=True)

# ---- cached model ----
@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = get_model()

# ---- helper text extraction functions ----
def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)

def extract_text_from_pdf(file_bytes: bytes, ocr_if_no_text: bool = True) -> str:
    text_pages = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_pages.append(page_text)
    except Exception:
        text_pages = []

    joined = "\n".join([p for p in text_pages if p and p.strip()])
    if joined.strip():
        return joined

    # fallback: OCR every page rendered as image
    if ocr_if_no_text:
        images = convert_from_bytes(file_bytes, dpi=200)
        ocr_text = []
        for img in images:
            ocr_page = pytesseract.image_to_string(img)
            ocr_text.append(ocr_page)
        return "\n".join([p for p in ocr_text if p and p.strip()])
    return ""

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode('utf-8', errors='ignore')
    except Exception:
        return str(file_bytes)

def extract_text_from_excel(file_bytes: bytes) -> Tuple[str, List[pd.DataFrame]]:
    try:
        xlsx = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
        # concatenate first few rows from sheets into text summary
        text_parts = []
        dfs = []
        for name, df in xlsx.items():
            dfs.append((name, df))
            snippet = f"Sheet: {name}\nColumns: {', '.join(map(str, df.columns[:10]))}\n"
            if not df.empty:
                snippet += df.head(3).to_csv(index=False)
            text_parts.append(snippet)
        return "\n".join(text_parts), dfs
    except Exception:
        return "", []

def extract_text_from_image(file_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    return pytesseract.image_to_string(img)

def extract_text(file) -> Dict[str, Any]:
    """
    Detect file type by name and content and extract:
    - text (string)
    - images (list of PIL images if applicable)
    - tables (list of dataframes for excel)
    """
    name = getattr(file, "name", "uploaded")
    raw = file.read()
    lower = name.lower()
    result = {"text": "", "images": [], "tables": []}

    try:
        if lower.endswith('.pdf'):
            result["text"] = extract_text_from_pdf(raw, ocr_if_no_text=True)
            # also extract page images for visual diff preview
            try:
                pages = convert_from_bytes(raw, dpi=120)
                result["images"] = pages[:4]  # limit to first 4
            except Exception:
                result["images"] = []
        elif lower.endswith('.docx'):
            result["text"] = extract_text_from_docx(raw)
        elif lower.endswith('.txt'):
            result["text"] = extract_text_from_txt(raw)
        elif lower.endswith('.xlsx') or lower.endswith('.xls'):
            text_summary, dfs = extract_text_from_excel(raw)
            result["text"] = text_summary
            result["tables"] = dfs
        elif lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            result["text"] = extract_text_from_image(raw)
            try:
                img = Image.open(io.BytesIO(raw)).convert('RGB')
                result["images"] = [img]
            except Exception:
                result["images"] = []
        else:
            # fallback try pdf, then txt
            result["text"] = extract_text_from_pdf(raw, ocr_if_no_text=True) or extract_text_from_txt(raw)
    except Exception as e:
        result["text"] = ""
    return result

# ---- utility comparators ----
def seq_text_diff(a: str, b: str, max_lines:int=200) -> Dict[str, Any]:
    """Produce a linewise diff and similarity ratio"""
    a_lines = [l for l in a.splitlines() if l.strip()]
    b_lines = [l for l in b.splitlines() if l.strip()]

    # limit size to avoid huge diffs
    if len(a_lines) > max_lines:
        a_lines = a_lines[:max_lines]
    if len(b_lines) > max_lines:
        b_lines = b_lines[:max_lines]

    matcher = SequenceMatcher(a="\n".join(a_lines), b="\n".join(b_lines))
    ratio = matcher.ratio()

    # line diffs (unified)
    diff = list(unified_diff(a_lines, b_lines, lineterm=""))
    return {"ratio": ratio, "diff_lines": diff}

def embed_similarity(a: str, b: str) -> float:
    """Compute semantic similarity between two blobs using sentence-transformers"""
    if not a.strip() or not b.strip():
        return 0.0
    try:
        emb_a = model.encode(a, convert_to_tensor=True)
        emb_b = model.encode(b, convert_to_tensor=True)
        sim = util.cos_sim(emb_a, emb_b).item()
        # clamp between 0 and 1
        return float(max(0.0, min(1.0, sim)))
    except Exception:
        return 0.0

def extract_numbers(text: str) -> List[str]:
    # capture numbers, currency-like, dates-like - simple regex
    nums = re.findall(r'\d[\d,./-]*\d|\d', text)
    # normalize commas
    nums = [n.replace(',', '') for n in nums]
    return nums

def numeric_diffs(a: str, b: str) -> Dict[str, Any]:
    na = set(extract_numbers(a))
    nb = set(extract_numbers(b))
    added = sorted(nb - na)
    removed = sorted(na - nb)
    common = sorted(na & nb)
    return {"added_numbers": added, "removed_numbers": removed, "common_numbers": common}

def compare_tables(tables_a: List[Any], tables_b: List[Any]) -> Dict[str, Any]:
    """
    Compare excel tables (list of (sheetname, df)) or list of dfs.
    We will return differences in sheet names, shapes and sample cell diffs.
    """
    summary = {"table_count_a": len(tables_a), "table_count_b": len(tables_b), "sheet_diffs": []}
    # normalize to list of tuples (name, df)
    ta = tables_a
    tb = tables_b
    # compare by index
    n = max(len(ta), len(tb))
    for i in range(n):
        row = {}
        if i < len(ta):
            name_a, df_a = ta[i]
            row["sheet_a"] = name_a
            row["shape_a"] = df_a.shape
            row["cols_a"] = list(map(str, df_a.columns.tolist()))
        else:
            row["sheet_a"] = None
            row["shape_a"] = None
            row["cols_a"] = []

        if i < len(tb):
            name_b, df_b = tb[i]
            row["sheet_b"] = name_b
            row["shape_b"] = df_b.shape
            row["cols_b"] = list(map(str, df_b.columns.tolist()))
        else:
            row["sheet_b"] = None
            row["shape_b"] = None
            row["cols_b"] = []

        # sample cell diffs (first 5 differing cells)
        diffs = []
        if i < len(ta) and i < len(tb):
            a_df = ta[i][1].fillna("").astype(str)
            b_df = tb[i][1].fillna("").astype(str)
            rows = min(a_df.shape[0], b_df.shape[0], 20)
            cols = min(a_df.shape[1], b_df.shape[1], 10)
            for r in range(rows):
                for c in range(cols):
                    va = a_df.iat[r, c]
                    vb = b_df.iat[r, c]
                    if va != vb:
                        diffs.append({"row": r, "col": c, "val_a": va, "val_b": vb})
                    if len(diffs) >= 8:
                        break
                if len(diffs) >= 8:
                    break
        row["cell_diffs_sample"] = diffs
        summary["sheet_diffs"].append(row)
    return summary

def compare_images(imgs_a: List[Image.Image], imgs_b: List[Image.Image]) -> Dict[str, Any]:
    """Compare first image of each list if available using imagehash and ssim"""
    report = {"image_count_a": len(imgs_a), "image_count_b": len(imgs_b)}
    if not imgs_a or not imgs_b:
        report["message"] = "One or both documents have no images for visual diff."
        return report

    # use first image from each (could be extended)
    a = imgs_a[0].convert('RGB')
    b = imgs_b[0].convert('RGB')

    # perceptual hash distance
    try:
        ha = imagehash.phash(a)
        hb = imagehash.phash(b)
        hamming = ha - hb
    except Exception:
        hamming = None

    # structural similarity: convert to grayscale numpy arrays, resize to match
    try:
        a_small = ImageOps.grayscale(a).resize((512, 512))
        b_small = ImageOps.grayscale(b).resize((512, 512))
        arr_a = np.array(a_small).astype(np.float32)
        arr_b = np.array(b_small).astype(np.float32)
        ssim_val = float(ssim(arr_a, arr_b))
    except Exception:
        ssim_val = None

    report.update({"phash_hamming": hamming, "ssim": ssim_val})
    return report

def aggregate_score(metrics: Dict[str, float]) -> float:
    """
    Combine text exact ratio, semantic sim, image sim (if present) into a single
    0-1 score. weights chosen heuristically.
    """
    txt = metrics.get("text_ratio", 0.0)
    sem = metrics.get("semantic_sim", 0.0)
    img = metrics.get("image_ssim", None)
    img_score = img if img is not None else 0.0

    # weight text exact less, semantic more
    score = (0.25 * txt) + (0.6 * sem) + (0.15 * img_score)
    # clamp
    return float(max(0.0, min(1.0, score)))

# ---- UI & main flow ----
# UI CSS (kept minimal and pleasant)
st.markdown("""
<style>
.report-box {background: rgba(255,255,255,0.04); padding: 1rem; border-radius:12px; border:1px solid rgba(255,255,255,0.06);}
.small-mono {font-family: monospace; font-size:0.9rem; color:#dcdcdc;}
</style>
""", unsafe_allow_html=True)

st.title("DiffPro AI — Full Comparison Engine")
st.write("Upload two documents (PDF, DOCX, TXT, XLSX, PNG/JPG). The app will extract text/images, run semantic + exact diffs, compare tables and images, and generate a JSON report.")

col_a, col_b = st.columns(2)
with col_a:
    file_a = st.file_uploader("Upload Document A", type=['pdf','docx','txt','xlsx','png','jpg','jpeg'], key="file_a")
with col_b:
    file_b = st.file_uploader("Upload Document B", type=['pdf','docx','txt','xlsx','png','jpg','jpeg'], key="file_b")

run = st.button("Run Full Comparison")

if run:
    if not (file_a and file_b):
        st.error("Please upload both files.")
    else:
        # Extract
        with st.spinner("Extracting content from documents..."):
            a = extract_text(file_a)
            # reset stream for file_b read by extract_text
            file_b.seek(0)
            b = extract_text(file_b)

        st.success("Extraction complete.")

        # Textual comparisons
        st.markdown("## Text Comparisons")
        text_a = a.get("text", "") or ""
        text_b = b.get("text", "") or ""

        with st.spinner("Running textual diffs and semantic similarity..."):
            seq = seq_text_diff(text_a, text_b)
            sem_sim = embed_similarity(text_a, text_b)
            num_diff = numeric_diffs(text_a, text_b)

        col1, col2, col3 = st.columns([1,1,1])
        col1.metric("Exact Text Similarity (difflib)", f"{seq['ratio']:.3f}")
        col2.metric("Semantic Similarity (embeddings)", f"{sem_sim:.3f}")
        # pick any number change metric
        added = num_diff.get("added_numbers", [])
        removed = num_diff.get("removed_numbers", [])
        col3.metric("Numeric Changes", f"+{len(added)} / -{len(removed)}")

        # Show top diff lines (first 30 lines)
        st.markdown("### Line-level Unified Diff (samples)")
        diff_sample = "\n".join(seq['diff_lines'][:200])
        st.code(diff_sample or "No line diffs (texts nearly identical).", language='text')

        # Show semantic highlights: top sentences most different
        st.markdown("### Semantic Highlights")
        if text_a.strip() and text_b.strip():
            # split into sentences and show sentences with low mutual similarity
            sent_a = nltk.tokenize.sent_tokenize(text_a)[:200]
            sent_b = nltk.tokenize.sent_tokenize(text_b)[:200]
            # embed sentences in batches
            try:
                emb_a = model.encode(sent_a, convert_to_tensor=True) if sent_a else None
                emb_b = model.encode(sent_b, convert_to_tensor=True) if sent_b else None

                # compute pairwise similarities for a small sample to find unique sentences
                sentence_pairs = []
                if emb_a is not None and emb_b is not None and len(sent_a) and len(sent_b):
                    sims = util.cos_sim(emb_a, emb_b).cpu().numpy()
                    # for each sentence in A, find max similarity in B, and show the ones with lowest max (i.e., unique in A)
                    max_sim_a = sims.max(axis=1)
                    low_idx_a = np.argsort(max_sim_a)[:5]
                    st.write("Top 5 sentences in Document A that are least matched in Document B:")
                    for idx in low_idx_a:
                        st.markdown(f"- {sent_a[idx][:300]}  — sim:{max_sim_a[idx]:.3f}")
                    max_sim_b = sims.max(axis=0)
                    low_idx_b = np.argsort(max_sim_b)[:5]
                    st.write("Top 5 sentences in Document B that are least matched in Document A:")
                    for idx in low_idx_b:
                        st.markdown(f"- {sent_b[idx][:300]}  — sim:{max_sim_b[idx]:.3f}")
            except Exception:
                st.write("Could not compute sentence-level highlights (embedding error).")
        else:
            st.write("One or both documents have empty extracted text.")

        # Table comparisons
        st.markdown("## Table / Excel Comparison")
        tables_a = a.get("tables", [])
        tables_b = b.get("tables", [])
        table_report = compare_tables(tables_a, tables_b)
        st.write(table_report)

        # Image comparisons
        st.markdown("## Visual (Image) Comparison")
        imgs_a = a.get("images", [])
        imgs_b = b.get("images", [])
        img_report = compare_images(imgs_a, imgs_b)
        st.write(img_report)

        # If images present, show preview side-by-side
        if imgs_a or imgs_b:
            st.markdown("### Image Previews (first page / first image)")
            pa, pb = st.columns(2)
            with pa:
                if imgs_a:
                    st.image(imgs_a[0], caption="Doc A — image preview", use_column_width=True)
                else:
                    st.info("No image in Document A")
            with pb:
                if imgs_b:
                    st.image(imgs_b[0], caption="Doc B — image preview", use_column_width=True)
                else:
                    st.info("No image in Document B")

        # Numeric diffs display
        st.markdown("## Numeric Field Differences (sample)")
        st.json(num_diff)

        # Combined metrics & aggregation
        metrics = {
            "text_ratio": seq.get("ratio", 0.0),
            "semantic_sim": sem_sim,
            "image_ssim": img_report.get("ssim", None)
        }
        combined = aggregate_score(metrics)
        st.markdown("## Combined Similarity Score")
        st.metric("Overall Similarity", f"{combined:.3f}")

        # Build JSON report for download
        st.markdown("## Download Comparison Report (JSON)")
        report = {
            "file_a": getattr(file_a, "name", "A"),
            "file_b": getattr(file_b, "name", "B"),
            "metrics": metrics,
            "combined_score": combined,
            "text_diff_lines": seq.get("diff_lines", [])[:1000],
            "numeric_diff": num_diff,
            "table_report": table_report,
            "image_report": img_report
        }
        report_json = json.dumps(report, indent=2)
        b64 = base64.b64encode(report_json.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="diffpro_report.json">Download report (JSON)</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Optional: show side-by-side text panes for manual inspection
        st.markdown("## Side-by-side Text View")
        left, right = st.columns(2)
        with left:
            st.markdown("### Document A — extracted text")
            st.text_area("Doc A Text", text_a[:20000], height=300)
        with right:
            st.markdown("### Document B — extracted text")
            st.text_area("Doc B Text", text_b[:20000], height=300)

        st.success("Full comparison finished. Use the JSON report to programmatically inspect results or to store evidence.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#999;'>DiffPro AI • Full Comparison Engine • Built with 🧠 + ❤️</div>", unsafe_allow_html=True)
