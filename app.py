import streamlit as st
import io, json, os, tempfile, nltk
from io import BytesIO
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

# Fix NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

st.set_page_config(page_title="Document Differentiator Pro", layout="wide", initial_sidebar_state="expanded")
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("🔍 Document Differentiator Pro")
st.markdown("**Compare PDFs, Word, Excel & Images** – OCR • Semantic Diff • Tables • Visual Similarity")

# Sidebar
st.sidebar.header("⚙️ Options")
use_ocr = st.sidebar.checkbox("Use OCR for scanned PDFs", value=True)
show_samples = st.sidebar.checkbox("Load Sample Files", value=False)

# Sample files (built-in for testing)
if show_samples:
    # Simple text samples
    sample_a = "Original Contract\nClause 1: Payment due in 30 days.\nClause 2: Termination notice required."
    sample_b = "Updated Contract\nClause 1: Payment due in 15 days.\nClause 2: Termination notice waived.\nNew Clause 3: Bonus added."
    
    # Save as temp files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_a)
        uploaded_a = open(f.name, 'rb')
        uploaded_a.name = 'original.txt'
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_b)
        uploaded_b = open(f.name, 'rb')
        uploaded_b.name = 'modified.txt'
    
    st.sidebar.success("Samples loaded! Use below.")

# Upload
col1, col2 = st.columns(2)
with col1:
    uploaded_a = st.file_uploader("📄 File A", type=['pdf', 'docx', 'txt', 'xlsx', 'png', 'jpg', 'jpeg'])
with col2:
    uploaded_b = st.file_uploader("📄 File B", type=['pdf', 'docx', 'txt', 'xlsx', 'png', 'jpg', 'jpeg'])

if st.button("🚀 Compare Documents", type="primary") and uploaded_a and uploaded_b:
    bytes_a, bytes_b = uploaded_a.read(), uploaded_b.read()
    name_a, name_b = uploaded_a.name, uploaded_b.name
    
    # Extraction function (from earlier)
    def extract_text(bytes_file, filename):
        name = filename.lower()
        if name.endswith('.txt'):
            return bytes_file.decode('utf-8'), 'txt'
        if name.endswith('.docx'):
            doc = docx.Document(BytesIO(bytes_file))
            return '\n'.join(p.text for p in doc.paragraphs), 'docx'
        if name.endswith('.pdf'):
            try:
                with pdfplumber.open(BytesIO(bytes_file)) as pdf:
                    text = '\n'.join(page.extract_text() or "" for page in pdf.pages)
                if len(text.strip()) > 50:
                    return text, 'pdf'
            except:
                pass
            images = convert_from_bytes(bytes_file, dpi=200)
            return '\n'.join(pytesseract.image_to_string(img) for img in images), 'pdf_ocr'
        if name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(bytes_file))
            return df.to_string(), 'excel'
        return "", 'unknown'
    
    # Image diff
    if name_a.lower().endswith(('png','jpg','jpeg')) and name_b.lower().endswith(('png','jpg','jpeg')):
        img_a = Image.open(BytesIO(bytes_a))
        img_b = Image.open(BytesIO(bytes_b))
        h_a = imagehash.phash(img_a)
        h_b = imagehash.phash(img_b)
        phash_sim = 1 - (h_a - h_b) / len(h_a.hash) ** 2
        g_a = np.array(img_a.convert('L'))
        g_b = np.array(img_b.convert('L'))
        ssim_score = ssim(g_a, g_b, data_range=g_a.max() - g_a.min())
        col1, col2, col3 = st.columns(3)
        col1.metric("pHash Similarity", f"{phash_sim:.1%}")
        col2.metric("SSIM Score", f"{ssim_score:.3f}")
        col3.metric("Status", "🟢 Similar" if phash_sim > 0.8 else "🔴 Different")
        st.image([img_a, img_b], caption=["File A", "File B"], width=300)
    
    else:
        text_a, type_a = extract_text(bytes_a, name_a)
        text_b, type_b = extract_text(bytes_b, name_b)
        
        # Line diff
        lines_a = text_a.splitlines()
        lines_b = text_b.splitlines()
        matcher = SequenceMatcher(None, lines_a, lines_b)
        ops = matcher.get_opcodes()
        changes = sum(1 for op in ops if op[0] != 'equal')
        line_sim = 1 - changes / max(len(lines_a), 1)
        
        # Semantic diff
        sents_a = [s for s in nltk.sent_tokenize(text_a) if s.strip()]
        sents_b = [s for s in nltk.sent_tokenize(text_b) if s.strip()]
        if sents_a and sents_b:
            emb_a = model.encode(sents_a, convert_to_tensor=True)
            emb_b = model.encode(sents_b, convert_to_tensor=True)
            sem_sim = util.cos_sim(emb_a, emb_b).diag().mean().item()
        else:
            sem_sim = 0
        
        col1, col2 = st.columns(2)
        col1.metric("Line Similarity", f"{line_sim:.1%}", delta=f"{line_sim*100:.0f}% match")
        col2.metric("Semantic Similarity", f"{sem_sim:.1%}", delta=f"{sem_sim*100:.0f}% meaning match")
        
        # Side-by-side
        st.subheader("📊 Side-by-Side Diff (First 100 Lines)")
        left, right = [], []
        for tag, i1, i2, j1, j2 in ops[:100]:
            if tag == 'equal':
                left.extend(lines_a[i1:i2])
                right.extend(lines_b[j1:j2])
            elif tag == 'delete':
                left.extend([f"🗑️ {x}" for x in lines_a[i1:i2]])
                right.extend([""] * (i2 - i1))
            elif tag == 'insert':
                left.extend([""] * (j2 - j1))
                right.extend([f"➕ {x}" for x in lines_b[j1:j2]])
            elif tag == 'replace':
                left.extend([f"✏️ {x}" for x in lines_a[i1:i2]])
                right.extend([f"✏️ {x}" for x in lines_b[j1:j2]])
        df_diff = pd.DataFrame({"File A": left[:100], "File B": right[:100]})
        st.dataframe(df_diff, use_container_width=True, hide_index=True)
        
        # Download report
        report = {"line_similarity": line_sim, "semantic_similarity": sem_sim, "diff_ops": ops}
        st.download_button("📥 Download JSON Report", json.dumps(report, indent=2), "diff_report.json", "json")

st.sidebar.markdown("---")
st.sidebar.markdown("[Built with ❤️ for your resume](https://streamlit.io)")
