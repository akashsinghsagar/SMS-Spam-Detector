
# 📧 SMS / Email Spam Detector

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)]()
[![NLP](https://img.shields.io/badge/NLP-NLTK-yellow.svg)]()
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)]()

A lightweight yet powerful **Machine Learning + NLP system** that classifies SMS and Email text as **Spam** or **Ham (Not Spam)** using TF-IDF and Multinomial Naive Bayes.

This real-time prediction system is deployed using **Streamlit Cloud** with a smooth and clean UI.

---

# 🚀 Live Demo

🔗 **Streamlit Web App:**
👉 [https://sms-spam-detector-akash.streamlit.app/](https://sms-spam-detector-akash.streamlit.app/)

---

# 🧠 Project Overview

This project includes:

✔ Fully automated NLP preprocessing
✔ TF-IDF text vectorization (3000 features)
✔ Highly accurate Multinomial Naive Bayes classifier
✔ Real-time spam prediction web application
✔ Complete training pipeline with dataset
✔ Easy-to-run code structure

---

# 🗂 Repository Structure

```
📦 SMS-Spam-Detector
│
├── app.py                        # Root-level wrapper for Streamlit Cloud
├── sms-spam-classifier/
│   ├── app.py                    # Main Streamlit application
│   ├── model.pkl                 # Trained ML model
│   ├── vectorizer.pkl            # Trained TF-IDF vectorizer
│   ├── spam.csv                  # Dataset used for training
│   ├── train_model.py            # Python script to train the model
│   ├── sms-spam-detection.ipynb  # Jupyter Notebook (EDA + Training)
│
├── requirements.txt              # All dependencies
└── README.md                     # Project documentation
```

---

# 🧹 NLP Preprocessing Pipeline

* Convert text to lowercase
* Tokenization using NLTK
* Remove stopwords
* Remove punctuation
* Apply Porter Stemming
* Return cleaned, stemmed text

```python
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)
```

---

# 🧠 Machine Learning Model

### **TF-IDF Vectorizer**

* `max_features = 3000`
* Converts text into numerical feature vectors

### **Multinomial Naive Bayes**

Chosen because:

* Excellent for text classification
* Fast training and prediction
* High accuracy and precision

---

# 📊 Model Performance

| Metric        | Score     |
| ------------- | --------- |
| **Accuracy**  | **0.98+** |
| **Precision** | **0.94+** |

Reliable & efficient for real-world SMS/Email spam detection.

---

# 🌐 Running the Project Locally

## **1️⃣ Clone the Repository**

```bash
git clone https://github.com/akashsinghsagar/SMS-Spam-Detector.git
cd SMS-Spam-Detector
```

## **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

## **3️⃣ Run Streamlit App**

```bash
streamlit run app.py
```

---

# 📉 Exploratory Data Analysis (EDA)

The notebook includes:

* Spam vs Ham distribution
* WordClouds (Spam & Ham)
* Message length analysis
* Frequent word analysis
* Correlation & patterns
* Training–validation accuracy

---

# 💾 Re-training the Model

To retrain the model using the dataset:

```bash
python sms-spam-classifier/train_model.py
```

This will regenerate:

* `model.pkl`
* `vectorizer.pkl`

---

<img width="1919" height="916" alt="Screenshot 2025-11-20 202248" src="https://github.com/user-attachments/assets/dfced8d6-c693-41b8-80c2-3cc1e9b928ef" />
<img width="1912" height="910" alt="Screenshot 2025-11-20 202314" src="https://github.com/user-attachments/assets/f601a85a-7ba2-4fef-b7b1-02b790c10ab1" />

---
# 🔮 Future Improvements

* Add deep learning model (LSTM / Bi-LSTM)
* Support for multiple languages
* Add email phishing detection
* Deploy via FastAPI REST API
* Modern UI upgrade for Streamlit

---

# 👨‍💻 Developed By

### **Akash Singh Sagar**

ML • NLP • Data Science • Python
Building practical, real-world AI applications.

