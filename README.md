# 📰 Fake News Detection using NLP & Machine Learning

A web application that detects whether a news article is **Fake** or **Real** using **Natural Language Processing (NLP)** and **Machine Learning**.

## 📌 Project Overview

The goal of this project is to automatically classify news articles into **Fake News** or **Real News** by analyzing the textual content. The project applies NLP preprocessing, TF-IDF feature extraction, and multiple machine learning algorithms to achieve high prediction accuracy.

---

## 🚀 Features

- Detects Fake or Real News
- NLP Text Preprocessing
- TF-IDF Vectorization
- Multiple Machine Learning Models
- Interactive Streamlit Web App
- Fast Prediction

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Pickle

---

## 📂 Dataset

This project uses the Fake News Dataset consisting of:

- Fake.csv
- True.csv

The dataset contains:

- News Title
- News Article
- Subject
- Label (Fake / Real)

---

## 🧹 NLP Pipeline

1. Convert text to lowercase
2. Remove URLs
3. Remove HTML tags
4. Remove punctuation
5. Remove numbers
6. Tokenization
7. Stop-word Removal
8. Lemmatization
9. Combine Title and Article
10. TF-IDF Vectorization

---

## 🤖 Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Decision Tree
- Random Forest

### Best Performing Model

**Random Forest Classifier**

Accuracy: **99.73%**

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## 📁 Project Structure

```
News-Project-NLP/
│
├── app.py
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
└── images/
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/dilip9045/News-Project-NLP.git
```

Go inside the project

```bash
cd News-Project-NLP
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🎯 Output

The application predicts whether the given news article is:

- ✅ Real News
- 🚨 Fake News

---

## 📈 Future Improvements

- BERT-based Fake News Detection
- Explainable AI (SHAP/LIME)
- News Source Verification
- Multilingual Fake News Detection
- Real-Time News API Integration

---

## 👨‍💻 Author

**Dilip Thakur**

GitHub: https://github.com/dilip9045

---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub.
