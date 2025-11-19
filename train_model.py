import os
import pickle
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ps = PorterStemmer()


def transform_text(text):
    text = str(text).lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    # cache stopwords to avoid repeated disk I/O
    if 'STOPWORDS' not in globals():
        global STOPWORDS
        STOPWORDS = set(stopwords.words('english'))
    for i in text:
        if i not in STOPWORDS and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


def load_dataset(path='spam.csv'):
    # try common column layouts
    df = pd.read_csv(path, encoding='latin-1')
    # common dataset has columns v1 (label) and v2 (text)
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    else:
        # try more generic names
        if 'label' in df.columns and 'text' in df.columns:
            df = df[['label', 'text']]
        else:
            # fallback: take first two columns
            cols = df.columns.tolist()
            df = df[[cols[0], cols[1]]].rename(columns={cols[0]: 'label', cols[1]: 'text'})
    return df


def main():
    print('Starting training script...')
    nltk.download('punkt')
    nltk.download('stopwords')

    if not os.path.exists('spam.csv'):
        raise SystemExit('spam.csv not found in current directory')

    df = load_dataset('spam.csv')
    # normalize label
    df['label'] = df['label'].astype(str).map(lambda x: 1 if x.lower().strip() in ('spam','1','true','yes') else 0)
    df['text'] = df['text'].astype(str).apply(transform_text)

    X = df['text']
    y = df['label']

    print('Vectorizing texts...')
    tfidf = TfidfVectorizer()
    X_vect = tfidf.fit_transform(X)

    print('Training classifier...')
    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print('Validation accuracy:', accuracy_score(y_test, preds))

    print('Saving artifacts...')
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print('Saved vectorizer.pkl and model.pkl')


if __name__ == '__main__':
    main()
