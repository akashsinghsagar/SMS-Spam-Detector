# 📧 SMS / Email Spam Classifier  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)]()
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)]()
[![NLTK](https://img.shields.io/badge/NLTK-Tokenizer-yellow)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

A modern machine-learning project to classify SMS/Email messages as **Spam** or **Ham (Not Spam)** using **TF-IDF**, **NLP preprocessing**, and **Multinomial Naive Bayes**.  
The complete model is deployed as a **Streamlit web application** for real-time predictions.

---

# 🚀 Features
- ✔ Clean & advanced NLP preprocessing  
- ✔ TF-IDF vectorizer (3000 features)  
- ✔ High-precision Multinomial Naive Bayes model  
- ✔ Full EDA with charts & word clouds  
- ✔ Streamlit web UI for live predictions  
- ✔ Saved model + vectorizer for deployment  

---

# 🗂 Project Structure
```

📦 SMS-Spam-Detector
│
├── app.py                     # Streamlit application
├── vectorizer.pkl             # Trained TF-IDF vectorizer
├── model.pkl                  # MultinomialNB trained model
├── sms-spam-detection.ipynb   # Full ML training notebook
├── spam.csv                   # Dataset
├── requirements.txt
└── README.md

````

---

# 🧠 Model Pipeline

### 🔹 **1. Text Preprocessing**
Includes:
- Lowercasing  
- Tokenization  
- Removing stopwords  
- Removing punctuation  
- Stemming using PorterStemmer  

Preprocessing function:

```python
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
````

---

### 🔹 **2. Feature Extraction**

TF-IDF Vectorizer with `max_features = 3000`:

```python
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
```

---

### 🔹 **3. Model Training**

Several ML models were tested:

| Algorithm               | Accuracy   | Precision  |
| ----------------------- | ---------- | ---------- |
| Multinomial Naive Bayes | **0.9806** | **0.9469** |
| SVM                     | Good       | Moderate   |
| Decision Tree           | Lower      | Lower      |
| Random Forest           | High       | High       |
| AdaBoost                | High       | Good       |
| XGBoost                 | High       | Great      |

🏆 **Multinomial Naive Bayes performed best overall.**

---

# 📊 Model Performance (Final Model)

| Metric        | Score                  |
| ------------- | ---------------------- |
| **Accuracy**  | **0.9806576402321083** |
| **Precision** | **0.946969696969697**  |

---

# 🌐 Run the Streamlit App

### 🔧 Install dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Start the app

```bash
streamlit run app.py
```

---

# 📉 Exploratory Data Analysis (EDA)

Included in the notebook:

* Class imbalance pie chart
* Word counts, sentence counts
* Histograms per class
* Pairplots
* Correlation heatmap
* WordClouds for Spam & Ham
* Top 30 frequent words (Spam & Ham)

---

# 💾 Saving the Model

```python
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))
```

---

# 🔮 Future Improvements

* Deploy on Streamlit Cloud / Render
* Add LSTM-based deep learning classifier
* Add support for multi-language SMS
* Improve UI/UX
* Add REST API using FastAPI

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Developed By

**Akash Singh Sagar**

