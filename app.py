import os
import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Ensure NLTK resources are available (Streamlit Cloud installs packages but not NLTK data)
def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

ensure_nltk_resources()

# cache stopwords
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = set()


def transform_text(text):
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    words = [t for t in tokens if t.isalnum()]
    words = [t for t in words if t not in STOPWORDS and t not in string.punctuation]
    stems = [ps.stem(t) for t in words]
    return " ".join(stems)

# Load artifacts relative to this file so Streamlit Cloud finds them correctly
BASE_DIR = os.path.dirname(__file__)
VECT_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

with open(VECT_PATH, 'rb') as f:
    tfidf = pickle.load(f)
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

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
