st.markdown("""
<style>

/* -------------------------------------------------- */
/* GLOBAL THEME */
/* -------------------------------------------------- */

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0A0A1F 0%, #1A1A3A 50%, #2D1B69 100%);
    color: #E8E8FF;
}

/* Remove scrollbar glitch */
.stApp::-webkit-scrollbar { width: 8px; }
.stApp::-webkit-scrollbar-thumb { background:#555; border-radius:4px; }

/* -------------------------------------------------- */
/* SIDEBAR FIXES */
/* -------------------------------------------------- */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #131324 0%, #0C0C18 100%);
    padding: 20px 10px !important;
}

.sidebar-title {
    text-align:center;
    margin-bottom:20px;
}

.sidebar-title h2 {
    color:#00D2FF;
    margin:0;
    font-size:1.8rem;
    font-weight:700;
}

.sidebar-title p {
    margin:0;
    color:#AAB4FF;
    font-size:0.9rem;
}

[data-testid="stSidebar"] .css-1cypcdb {  /* radio buttons */
    padding-left:10px !important;
}
[data-testid="stSidebar"] label {
    font-size:1rem !important;
    color:#d0d0ff !important;
}

/* -------------------------------------------------- */
/* BUTTON FIX */
/* -------------------------------------------------- */

.stButton > button {
    background: linear-gradient(45deg, #00D2FF, #7C4DFF);
    color: white !important;
    border: none;
    border-radius: 50px;
    padding: 14px 36px;
    font-size: 1.1rem;
    font-weight: 600;
    box-shadow:0 6px 18px rgba(0,0,0,0.35);
    transition:0.25s;
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow:0 10px 25px rgba(0,0,0,0.45);
}

/* -------------------------------------------------- */
/* GLASS CARD FIX */
/* -------------------------------------------------- */

.card {
    background: rgba(255,255,255,0.05);
    padding: 1.8rem;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    margin-bottom: 25px;
}

/* -------------------------------------------------- */
/* BEAUTIFUL DIFF VIEWER */
/* -------------------------------------------------- */

table.diff {
    width: 100%;
    border-collapse: collapse;
    font-family: monospace;
    font-size: 0.95rem;
    margin-top: 20px;
}

table.diff th {
    background: rgba(255, 255, 255, 0.05);
    color: #F0F0FF;
    padding: 6px;
    border: 1px solid rgba(255,255,255,0.08);
    font-weight:600;
}

table.diff td {
    padding: 6px;
    vertical-align: top;
    border: 1px solid rgba(255,255,255,0.03);
}

.diff_header {
    background: rgba(0,0,0,0.2);
    color:#FFF;
}

.diff_next { background: transparent; }

/* Added */
.diff_add {
    background-color: rgba(76, 175, 80, 0.28) !important; 
}
/* Removed */
.diff_sub {
    background-color: rgba(244, 67, 54, 0.28) !important; 
}
/* Changed */
.diff_chg {
    background-color: rgba(255, 202, 40, 0.28) !important; 
}

/* -------------------------------------------------- */
/* ABOUT PAGE FIX */
/* -------------------------------------------------- */

.about-container {
    display:flex;
    gap:2rem;
    align-items:center;
}

.about-img {
    width:180px;
    height:180px;
    border-radius:50%;
    border:3px solid #00D2FF;
    box-shadow:0 0 20px rgba(0,210,255,0.4);
}

.about-text {
    font-size:1.2rem;
    line-height:1.7;
    color:#EEE;
}

.about-text a {
    color:#7C4DFF;
    font-weight:600;
}

/* -------------------------------------------------- */
/* SMALL MONO TEXT */
/* -------------------------------------------------- */

.small-mono {
    font-family: monospace;
    font-size: 0.88rem;
    color: #CACACA;
}

</style>
""", unsafe_allow_html=True)
