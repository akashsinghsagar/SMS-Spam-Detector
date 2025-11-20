import os
import streamlit as st
import pickle
import string
import nltk
from nltk.stem import PorterStemmer
import importlib

ps = PorterStemmer()

# --- Ensure NLTK Data Works on Streamlit Cloud ---
# Streamlit only allows writing to /tmp, so we force downloads there
nltk.data.path.append('/tmp')

def ensure_nltk_resources():
    resources = {
        'tokenizers/punkt': 'punkt',
        'corpora/stopwords': 'stopwords',
        'tokenizers/punkt_tab': 'punkt_tab'   # Required in latest NLTK versions
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, download_dir='/tmp')

ensure_nltk_resources()

# --- Load Stopwords ---
try:
    corp = importlib.import_module('nltk.corpus')
    STOPWORDS = set(corp.stopwords.words('english'))
except Exception:
    STOPWORDS = set()


# --- Preprocessing Function ---
def transform_text(text):
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    words = [t for t in tokens if t.isalnum()]
    words = [t for t in words if t not in STOPWORDS and t not in string.punctuation]
    stems = [ps.stem(t) for t in words]
    return " ".join(stems)


# --- Load Model + Vectorizer ---
BASE_DIR = os.path.dirname(__file__)
VECT_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

with open(VECT_PATH, 'rb') as f:
    tfidf = pickle.load(f)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


# --- Streamlit UI ---
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if int(result) == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
