import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download once
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words('english'))

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")

st.title("💬 Sentiment Analysis Classifier")
st.markdown(
    """
    <style>
    .stTextArea textarea {font-size: 18px;}
    .result-box {
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("Enter a text or sentence below to predict its **emotion/sentiment**:")

user_input = st.text_area("Your text:", height=150)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned_text = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized)[0]
        
        # Color-coding results
        if prediction == 0:
            color = "#FFCCCB"  # red/pink
            sentiment = "😞 Negative"
        elif prediction == 1:
            color = "#F9E79F"  # yellow
            sentiment = "😐 Neutral"
        else:
            color = "#ABEBC6"  # green
            sentiment = "😊 Positive"

        st.markdown(
            f'<div class="result-box" style="background-color:{color};">{sentiment}</div>',
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")
