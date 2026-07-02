import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Download required NLTK data
# -----------------------------
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# -----------------------------
# Load Model & TF-IDF Vectorizer
# -----------------------------
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# -----------------------------
# NLP Preprocessing
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation & numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenization
    words = nltk.word_tokenize(text)

    # Stopword removal + Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(words)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection")
st.write("Detect whether a news article is **Fake** or **Real** using NLP and Machine Learning.")

st.sidebar.title("About")
st.sidebar.info("""
Project: Fake News Detection

Algorithms Compared:
- Logistic Regression
- Naive Bayes
- Linear SVM
- Decision Tree
- Random Forest

Best Model: Random Forest
""")

title = st.text_input("Enter News Title")

article = st.text_area(
    "Enter News Content",
    height=250
)

if st.button("Predict"):

    if title.strip() == "" and article.strip() == "":
        st.warning("Please enter a news title or article.")
    else:

        combined_text = title + " " + article

        processed = preprocess_text(combined_text)

        vector = tfidf.transform([processed])

        prediction = model.predict(vector)[0]

        st.markdown("---")

        if prediction == 0:
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")

        # Show confidence if supported
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(vector).max() * 100
            st.info(f"Confidence: {confidence:.2f}%")