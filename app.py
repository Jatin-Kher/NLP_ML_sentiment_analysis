import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# ---- NLTK Setup ----
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ---- Streamlit Page Config ----
st.set_page_config(
    page_title="Sentiment Emotion Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom Styling ----
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stTextArea textarea {
    border-radius: 10px !important;
    border: 2px solid #e0e0e0 !important;
    font-size: 16px !important;
    transition: all 0.3s ease;
}
.stTextArea textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 10px rgba(102, 126, 234, 0.3) !important;
}
.stButton button {
    width: 100%;
    padding: 15px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    cursor: pointer !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}
.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}
.emoji-icon {
    font-size: 60px;
    text-align: center;
    margin: 10px 0;
}
.emotion-badge {
    display: inline-block;
    padding: 20px 40px;
    border-radius: 15px;
    color: white;
    font-size: 28px;
    font-weight: 800;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 2px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    margin: 20px 0;
}
.emotion-sadness { background: linear-gradient(135deg, #667eea 0%, #4c63d2 100%); }
.emotion-anger { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
.emotion-love { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
.emotion-surprise { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
.emotion-fear { background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); }
.emotion-joy { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
.info-box {
    background: #f8f9fa;
    border-left: 5px solid #667eea;
    padding: 20px;
    border-radius: 8px;
    margin: 15px 0;
    font-size: 14px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ---- Load Model and Vectorizer ----
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)

model, vectorizer, error = load_model_artifacts()

if error:
    st.error(f"❌ {error}")
    st.info("Make sure `sentiment_model.pkl` and `vectorizer.pkl` exist in the current directory.")
    st.stop()

# ---- Emotion Mapping ----
emotion_mapping = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

emotion_emojis = {
    "sadness": "😢",
    "anger": "😠",
    "love": "❤️",
    "surprise": "😲",
    "fear": "😨",
    "joy": "😊"
}

emotion_descriptions = {
    "sadness": "This text conveys sadness, grief, or melancholy.",
    "anger": "This text conveys anger, frustration, or irritation.",
    "love": "This text conveys love, affection, or warmth.",
    "surprise": "This text conveys surprise, amazement, or astonishment.",
    "fear": "This text conveys fear, anxiety, or concern.",
    "joy": "This text conveys joy, happiness, or delight."
}

# ---- Text Preprocessing ----
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([char for char in text if not char.isdigit()])
    text = ''.join([char for char in text if ord(char) < 128])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# ---- UI Layout ----
st.markdown('<div class="header-title"><h1 style="text-align:center;">🎭 Sentiment Emotion Detector</h1></div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<strong>✨ How it works:</strong> Enter any text and our trained NLP model will analyze it to detect 
the underlying emotion: <b>Sadness</b>, <b>Anger</b>, <b>Love</b>, <b>Surprise</b>, <b>Fear</b>, or <b>Joy</b>.
</div>
""", unsafe_allow_html=True)

# ---- Input Section ----
st.subheader("📝 Enter Your Text")
user_input = st.text_area(
    label="Type something to analyze...",
    height=150,
    placeholder="Example: I'm feeling so happy today! Or I'm really upset about what happened.",
    help="The model will analyze this text to predict the emotion"
)

# ---- Prediction Button ----
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

# ---- Prediction Logic ----
if analyze_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze!")
    else:
        with st.spinner("🤔 Analyzing sentiment..."):
            try:
                cleaned_text = preprocess_text(user_input)
                if not cleaned_text.strip():
                    st.warning("⚠️ Text became empty after preprocessing. Try entering different text.")
                else:
                    vectorized = vectorizer.transform([cleaned_text])
                    prediction = model.predict(vectorized)[0]
                    emotion = emotion_mapping.get(int(prediction), "unknown")
                    emoji = emotion_emojis.get(emotion, "❓")
                    description = emotion_descriptions.get(emotion, "Unknown emotion")

                    st.markdown("---")
                    st.markdown("## ✅ Prediction Result")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(f'<div class="emoji-icon">{emoji}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="emotion-badge emotion-{emotion}">{emotion}</div>', unsafe_allow_html=True)
                    st.info(f"💭 {description}")

                    with st.expander("📊 Detailed Analysis"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Original Text:**")
                            st.code(user_input, language="text")
                        with col2:
                            st.write("**Cleaned Text (after preprocessing):**")
                            st.code(cleaned_text, language="text")
                        st.write("**Model Output:**")
                        st.code(f"Prediction value: {int(prediction)}\nEmotion: {emotion}", language="text")
                        st.write("**Vectorizer Info:**")
                        st.write(f"- Input shape: {vectorized.shape}")
                        st.write(f"- Vocabulary size: {len(vectorizer.get_feature_names_out())}")
                    st.markdown("---")
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("Please check that your model files are compatible with your preprocessing steps.")

# ---- Footer ----
st.markdown("---")
st.markdown("### 📚 Example Texts to Try")

example_col1, example_col2, example_col3 = st.columns(3)
with example_col1:
    if st.button("😊 Joy Example"):
        st.session_state.example_text = "I'm so happy and excited about my new job!"
with example_col2:
    if st.button("😢 Sadness Example"):
        st.session_state.example_text = "I feel really sad and disappointed about how things turned out."
with example_col3:
    if st.button("😠 Anger Example"):
        st.session_state.example_text = "I'm so furious and angry about what you did to me!"

st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
<p style="font-size: 14px; color: #666;">
<strong>🤖 Built with:</strong> Streamlit | Scikit-Learn | NLTK | Logistic Regression<br>
<strong>📊 Model:</strong> Trained on sentiment analysis dataset with TF-IDF vectorization<br>
<strong>✨ Emotions:</strong> Sadness, Anger, Love, Surprise, Fear, Joy
</p>
</div>
""", unsafe_allow_html=True)
