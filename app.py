import os
import streamlit as st
import pickle
import string
import nltk
from nltk.stem import PorterStemmer
import importlib

# =============================
#        CONFIG & STYLING
# =============================
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium UI
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .title {
        font-size: 40px !important;
        font-weight: 700 !important;
        text-align: center;
        padding-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #b3b3b3;
        margin-bottom: 25px;
    }
    .result-card {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        text-align: center;
        font-size: 28px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

ps = PorterStemmer()

# =============================
#  NLTK Setup for Streamlit Cloud
# =============================
nltk.data.path.append('/tmp')

def ensure_nltk_resources():
    resources = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'tokenizers/punkt_tab': 'punkt_tab'
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, download_dir='/tmp')

ensure_nltk_resources()

# Load stopwords safely
try:
    corp = importlib.import_module('nltk.corpus')
    STOPWORDS = set(corp.stopwords.words('english'))
except:
    STOPWORDS = set()

# =============================
#     PREPROCESSING
# =============================
def transform_text(text):
    text = str(text).lower().strip()
    tokens = nltk.word_tokenize(text)
    words = [t for t in tokens if t.isalnum()]
    words = [t for t in words if t not in STOPWORDS]
    stems = [ps.stem(t) for t in words]
    return " ".join(stems)

# =============================
#     LOAD MODEL & VECTORIZER
# =============================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
    tfidf = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

# =============================
#        SIDEBAR
# =============================
st.sidebar.title("⚙️ About App")
st.sidebar.info("""
This is a **Machine Learning–powered Spam Detector** that classifies messages as  
**Spam** or **Not Spam** in real-time using:
- TF-IDF Vectorizer  
- Naive Bayes Model  
- Custom text preprocessing  
""")

st.sidebar.write("👨‍💻 *By Akash Singh Sagar*")

# =============================
#            UI
# =============================
st.markdown("<div class='title'>📩 Email / SMS Spam Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Instant ML-powered message classification</div>", unsafe_allow_html=True)

input_sms = st.text_area("✍️ Enter your message below")

if st.button("🚀 Predict", use_container_width=True):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        transformed = transform_text(input_sms)
        vector_input = tfidf.transform([transformed])
        result = int(model.predict(vector_input)[0])

        if result == 1:
            st.markdown(
                "<div class='result-card' style='background-color:#ff4d4d;color:white;'>🚨 SPAM DETECTED</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='result-card' style='background-color:#00c853;color:white;'>✅ NOT SPAM</div>",
                unsafe_allow_html=True,
            )

