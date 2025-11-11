import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---- NLTK downloads (quiet) ----
nltk.download('punkt', quiet=True)
nltk.download('punkt')
nltk.download('stopwords', quiet=True)

# ---- Load model & vectorizer safely ----
@st.cache_resource
def load_model_and_vectorizer(model_path="sentiment_model.pkl", vec_path="vectorizer.pkl"):
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)

model, vectorizer, load_error = load_model_and_vectorizer()

if load_error:
    st.error(f"Error loading model/vectorizer: {load_error}")
    st.stop()

# ---- stop words set ----
stop_words = set(stopwords.words('english'))

# ---- preprocessing: must match training preprocessing exactly ----
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # remove punctuation, but keep spaces
    text = text.translate(str.maketrans("", "", string.punctuation))
    # tokenize (optional - depends on how you trained the vectorizer)
    words = word_tokenize(text)
    # remove stopwords and leftover punctuation tokens, keep lowercase
    words = [w for w in words if w.isalpha() and w not in stop_words]
    return " ".join(words)

# ---- UI layout ----
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")
st.title("💬 Sentiment Analysis Classifier — Debug Mode")

st.write("Model classes (as learned by model):")
st.write(getattr(model, "classes_", "Model has no attribute classes_"))

# Provide a mapping from model output to human-friendly label.
# Update this mapping to match how you encoded targets during training.
# Example mapping for 0..5 (change as needed):
default_mapping = {
    0: "anger/negative",
    1: "fear/neutral",
    2: "joy/positive",
    3: "sadness/negative",
    4: "surprise/positive",
    5: "love/positive"
}
st.info("Update label mapping if your labels differ from default.")
label_mapping = st.text_area("Label mapping as Python dict (edit if needed):",
                             value=str(default_mapping), height=130)

try:
    label_mapping = eval(label_mapping)
except Exception as e:
    st.error("Invalid mapping dict. Revert to JSON-like Python dict. Error: " + str(e))
    st.stop()

# text input
user_input = st.text_area("Enter text to analyze", height=140)

# show preprocessing details in an expander
with st.expander("Show preprocessing & debug info"):
    st.write("Raw input:")
    st.write(user_input)
    cleaned = clean_text(user_input if user_input else "")
    st.write("Cleaned input (what will be vectorized):")
    st.write(cleaned)
    if vectorizer is not None:
        try:
            X_sample = vectorizer.transform([cleaned])
            st.write("Vectorized shape (rows, cols):", X_sample.shape)
            # Show number of non-zero features in this sample
            st.write("Non-zero features in sample:", X_sample.nnz)
            # Show vocabulary size of vectorizer (if available)
            vocab_size = getattr(vectorizer, "vocabulary_", None)
            if vocab_size is not None:
                st.write("Vectorizer vocabulary size:", len(vocab_size))
            else:
                # For TF-IDF or other vectorizers:
                try:
                    st.write("Feature length via shape of transform: ", X_sample.shape[1])
                except:
                    pass
        except Exception as e:
            st.error("Error during vectorizer.transform: " + str(e))

# Predict button
if st.button("Analyze Sentiment"):
    if not user_input or user_input.strip() == "":
        st.warning("Please enter text to analyze.")
    else:
        cleaned_text = clean_text(user_input)
        try:
            vec = vectorizer.transform([cleaned_text])
        except Exception as e:
            st.error("Error transforming text with vectorizer: " + str(e))
            st.stop()

        try:
            pred = model.predict(vec)
            pred_proba = None
            if hasattr(model, "predict_proba"):
                pred_proba = model.predict_proba(vec)
        except Exception as e:
            st.error("Model prediction error: " + str(e))
            st.stop()

        label = pred[0]
        human_label = label_mapping.get(label, f"label_{label}")

        # show result
        st.markdown("### Prediction")
        st.write("Numeric label:", int(label))
        st.write("Mapped label:", human_label)
        if pred_proba is not None:
            st.write("Predicted probabilities (first row):")
            st.write(pred_proba[0])

        # colored output box
        color = "#FFFFFF"
        if "neg" in human_label.lower() or "anger" in human_label.lower() or "sad" in human_label.lower():
            color = "#FFCCCB"
        elif "neu" in human_label.lower() or "fear" in human_label.lower():
            color = "#F9E79F"
        else:
            color = "#ABEBC6"

        st.markdown(f'<div style="padding:12px;border-radius:8px;background:{color};font-weight:bold;text-align:center">{human_label}</div>', unsafe_allow_html=True)
