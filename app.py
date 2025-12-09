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

# ============ NLTK FIX ============
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ============ MODEL ============
model = SentenceTransformer('all-MiniLM-L6-v2')

# ============ BEAUTIFUL CSS ============
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #4361ee; color: white; border-radius: 8px; padding: 10px 20px;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    h1, h2, h3 {color: #4361ee;}
    .css-1d391kg {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# ============ PAGE CONFIG & NAVIGATION ============
st.set_page_config(page_title="Document Diff Pro", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home - Compare Files", "About the App", "Developer"])

# ============ PAGE 1: HOME ============
if page == "Home - Compare Files":
    st.title("Document Differentiator Pro")
    st.markdown("### Compare any two documents instantly — PDF, Word, Excel, Images & Scanned files supported!")

    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Upload File A", type=['pdf','docx','txt','xlsx','png','jpg','jpeg'])
    with col2:
        file_b = st.file_uploader("Upload File B", type=['pdf','docx','txt','xlsx','png','jpg','jpeg'])

    if file_a and file_b:
        with st.spinner("Analyzing documents..."):
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
                col1.metric("Visual Similarity (pHash)", f"{phash_sim:.1%}")
                col2.metric("Structural Similarity (SSIM)", f"{ssim_score:.3f}")
                st.image([img_a, img_b], caption=["File A", "File B"], width=350)

            else:
                def extract_text(data, filename):
                    name = filename.lower()
                    if name.endswith('.txt'): return data.decode('utf-8'), 'txt'
                    if name.endswith('.docx'):
                        doc = docx.Document(io.BytesIO(data))
                        return "\n".join(p.text for p in doc.paragraphs), 'docx'
                    if name.endswith('.pdf'):
                        try:
                            with pdfplumber.open(io.BytesIO(data)) as pdf:
                                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                            if len(text.strip()) > 50: return text, 'pdf'
                        except: pass
                        images = convert_from_bytes(data, dpi=200)
                        return "\n".join(pytesseract.image_to_string(img) for img in images), 'pdf_ocr'
                    if name.endswith(('.xlsx','.xls')):
                        return pd.read_excel(io.BytesIO(data)).to_string(), 'excel'
                    return "", 'unknown'

                text_a, _ = extract_text(bytes_a, name_a)
                text_b, _ = extract_text(bytes_b, name_b)

                lines_a = text_a.splitlines()
                lines_b = text_b.splitlines()
                matcher = SequenceMatcher(None, lines_a, lines_b)
                ops = matcher.get_opcodes()
                changes = sum(1 for op in ops if op[0] != 'equal')
                line_sim = 1 - changes / max(len(lines_a), 1)

                sents_a = [s for s in nltk.sent_tokenize(text_a) if s.strip()]
                sents_b = [s for s in nltk.sent_tokenize(text_b) if s.strip()]
                sem_sim = 0
                if sents_a and sents_b:
                    emb_a = model.encode(sents_a, convert_to_tensor=True)
                    emb_b = model.encode(sents_b, convert_to_tensor=True)
                    sem_sim = util.cos_sim(emb_a, emb_b).diag().mean().item()

                col1, col2 = st.columns(2)
                col1.metric("Line-by-Line Match", f"{line_sim:.1%}")
                col2.metric("Semantic (Meaning) Match", f"{sem_sim:.1%}")

                st.subheader("Side-by-Side Comparison")
                left, right = [], []
                for tag, i1, i2, j1, j2 in ops[:100]:
                    a_part = lines_a[i1:i2]
                    b_part = lines_b[j1:j2]
                    max_len = max(len(a_part), len(b_part))
                    a_part += [""] * (max_len - len(a_part))
                    b_part += [""] * (max_len - len(b_part))
                    for a, b in zip(a_part, b_part):
                        if tag == 'equal':
                            left.append(a); right.append(b)
                        elif tag == 'delete':
                            left.append(f"[-] {a}"); right.append("")
                        elif tag == 'insert':
                            left.append(""); right.append(f"[+] {b}")
                        elif tag == 'replace':
                            left.append(f"[~] {a}"); right.append(f"[~] {b}")

                df = pd.DataFrame({"File A": left, "File B": right})
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.download_button("Download Full Report", json.dumps({"line_sim": line_sim, "semantic_sim": sem_sim}, indent=2), "report.json")

# ============ PAGE 2: ABOUT ============
elif page == "About the App":
    st.title("About Document Differentiator Pro")
    st.markdown("""
    This is a **powerful AI-powered document comparison tool** built with Python and Streamlit.

    ### Features
    - Compare **PDF, Word, Excel, Images**
    - **OCR** for scanned documents
    - **Semantic comparison** using BERT (understands meaning)
    - **Visual diff** for images (pHash + SSIM)
    - Clean side-by-side view
    - Downloadable reports

    Perfect for lawyers, auditors, researchers, and students!
    """)

# ============ PAGE 3: DEVELOPER ============
elif page == "Developer":
    st.title("Developer")
    st.image("https://via.placeholder.com/150", width=150)
    st.markdown("""
    **Name:** Indu Reddy  
    **Location:** Bengaluru, India  
    **Role:** Data Analyst Intern

    Passionate about building intelligent tools that solve real problems.

    **Skills:** Python • Streamlit • Machine Learning • NLP • Computer Vision • Cloud Deployment

    **GitHub:** [github.com/indureddy20](https://github.com/indureddy20)  
    **LinkedIn:** (linkedin.com/in/indu-priya-mapakshi-871336241)


    """)

st.sidebar.success("The Project differentiates your documents at one go")
