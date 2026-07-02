import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===========================
# Download NLTK Resources
# ===========================
resources = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
]

for path, resource in resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

# ===========================
# Load Model & TF-IDF
# ===========================
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# ===========================
# NLP Components
# ===========================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ===========================
# Text Preprocessing
# ===========================
def preprocess_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation & numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    words = nltk.word_tokenize(text)

    # Stopword Removal + Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(words)

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection using NLP & Machine Learning")

st.markdown("""
Enter a **news title** and **news article** below to predict whether it is **Fake** or **Real**.
""")

title = st.text_input("News Title")

article = st.text_area(
    "News Article",
    height=250
)

if st.button("Predict"):

    if title.strip() == "" and article.strip() == "":
        st.warning("Please enter a news title or article.")
        st.stop()

    combined_text = title + " " + article

    processed = preprocess_text(combined_text)

    vector = tfidf.transform([processed])

    prediction = model.predict(vector)[0]

    st.markdown("---")

    if prediction == 0:
        st.error("🚨 Prediction: Fake News")
    else:
        st.success("✅ Prediction: Real News")

    # Confidence Score
    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(vector).max() * 100
        st.info(f"Confidence: {confidence:.2f}%")

st.markdown("---")
st.caption("Developed using Natural Language Processing (NLP), TF-IDF, and Machine Learning.")