import streamlit as st
import io
import json
import nltk
import pandas as pd
import numpy as np
from PIL import Image
import pdfplumber
import docx
from pdf2image import convert_from_bytes
import pytesseract
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import imagehash
from skimage.metrics import structural_similarity as ssim

# Fix NLTK (must be first)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

st.set_page_config(page_title="Document Differentiator Pro", layout="wide")
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("Document Differentiator Pro")
st.markdown("**Compare PDFs, Word, Excel & Images** — OCR • Semantic • Tables • Visual Diff")

# Sidebar
st.sidebar.header("Options")
use_samples = st.sidebar.checkbox("Use Built-in Sample Files", value=False)

# Built-in samples (fixed — no file handle issues)
if use_samples:
    sample_text_a = "Original Contract\nPayment due in 30 days\nTermination requires 60-day notice\nBonus: 5%"
    sample_text_b = "Updated Contract\nPayment due in 15 days\nNo termination notice needed\nBonus: 10%\nNew clause added"
    
    # Create fake file-like objects
    a_file = io.BytesIO(sample_text_a.encode('utf-8'))
    a_file.name = "original.txt"
    b_file = io.BytesIO(sample_text_b.encode('utf-8'))
    b_file.name = "modified.txt"
    
    st.sidebar.success("Sample files loaded!")
else:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_a = st.file_uploader("File A", type=['pdf','docx','txt','xlsx','png','jpg','jpeg'])
    with col2:
        uploaded_b = st.file_uploader("File B", type=['pdf','docx','txt','xlsx','png','jpg','jpeg'])

if st.button("Compare Documents", type="primary"):
    if use_samples:
        file_a, file_b = a_file, b_file
    else:
        if not uploaded_a or not uploaded_b:
            st.error("Please upload both files!")
            st.stop()
        file_a, file_b = uploaded_a, uploaded_b

    bytes_a = file_a.read()
    bytes_b = file_b.read()
    name_a = file_a.name
    name_b = file_b.name

    # Image comparison
    if name_a.lower().endswith(('png','jpg','jpeg')) and name_b.lower().endswith(('png','jpg','jpeg')):
        img_a = Image.open(io.BytesIO(bytes_a))
        img_b = Image.open(io.BytesIO(bytes_b))
        h1 = imagehash.phash(img_a)
        h2 = imagehash.phash(img_b)
        phash_sim = 1 - (h1 - h2) / len(h1.hash)**2
        g1 = np.array(img_a.convert('L'))
        g2 = np.array(img_b.convert('L'))
        ssim_score = ssim(g1, g2, data_range=255)
        col1, col2 = st.columns(2)
        col1.metric("pHash Similarity", f"{phash_sim:.1%}")
        col2.metric("SSIM Score", f"{ssim_score:.3f}")
        st.image([img_a, img_b], caption=["File A", "File B"], width=350)

    else:
        # Text extraction
        def extract_text(data, filename):
            name = filename.lower()
            if name.endswith('.txt'):
                return data.decode('utf-8'), 'txt'
            if name.endswith('.docx'):
                doc = docx.Document(io.BytesIO(data))
                return "\n".join(p.text for p in doc.paragraphs), 'docx'
            if name.endswith('.pdf'):
                try:
                    with pdfplumber.open(io.BytesIO(data)) as pdf:
                        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                    if len(text.strip()) > 50:
                        return text, 'pdf'
                except: pass
                images = convert_from_bytes(data, dpi=200)
                return "\n".join(pytesseract.image_to_string(img) for img in images), 'pdf_ocr'
            if name.endswith(('.xlsx','.xls')):
                df = pd.read_excel(io.BytesIO(data))
                return df.to_string(), 'excel'
            return "", 'unknown'

        text_a, _ = extract_text(bytes_a, name_a)
        text_b, _ = extract_text(bytes_b, name_b)

        lines_a = text_a.splitlines()
        lines_b = text_b.splitlines()
        matcher = SequenceMatcher(None, lines_a, lines_b)
        ops = matcher.get_opcodes()

        # Line similarity
        changes = sum(1 for op in ops if op[0] != 'equal')
        line_sim = 1 - changes / max(len(lines_a), 1)

        # Semantic similarity
        sents_a = [s for s in nltk.sent_tokenize(text_a) if s.strip()]
        sents_b = [s for s in nltk.sent_tokenize(text_b) if s.strip()]
        sem_sim = 0
        if sents_a and sents_b:
            emb_a = model.encode(sents_a, convert_to_tensor=True)
            emb_b = model.encode(sents_b, convert_to_tensor=True)
            sem_sim = util.cos_sim(emb_a, emb_b).diag().mean().item()

        col1, col2 = st.columns(2)
        col1.metric("Line-by-Line Match", f"{line_sim:.1%}")
        col2.metric("Meaning (Semantic) Match", f"{sem_sim:.1%}")

        # Side-by-side diff (FIXED: equal length)
        st.subheader("Side-by-Side Comparison")
        left_lines = []
        right_lines = []
        for tag, i1, i2, j1, j2 in ops[:100]:
            a_part = lines_a[i1:i2]
            b_part = lines_b[j1:j2]
            max_len = max(len(a_part), len(b_part))
            a_part += [""] * (max_len - len(a_part))
            b_part += [""] * (max_len - len(b_part))
            for a, b in zip(a_part, b_part):
                if tag == 'equal':
                    left_lines.append(a)
                    right_lines.append(b)
                elif tag == 'delete':
                    left_lines.append(f"[-] {a}")
                    right_lines.append("")
                elif tag == 'insert':
                    left_lines.append("")
                    right_lines.append(f"[+] {b}")
                elif tag == 'replace':
                    left_lines.append(f"[~] {a}")
                    right_lines.append(f"[~] {b}")

        df_diff = pd.DataFrame({"File A": left_lines, "File B": right_lines})
        st.dataframe(df_diff, use_container_width=True, hide_index=True)

        # Download report
        report = {"line_similarity": line_sim, "semantic_similarity": sem_sim}
        st.download_button("Download Report", json.dumps(report, indent=2), "diff_report.json")

st.sidebar.markdown("---")
st.sidebar.success("Live & Permanent!")
st.sidebar.markdown("Your resume project is ready!")
