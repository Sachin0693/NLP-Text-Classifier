import streamlit as st
import joblib
import numpy as np

# --- Load model and vectorizer ---
model = joblib.load("logreg_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Page Configuration ---
st.set_page_config(
    page_title="NLP Text Classifier",
    page_icon="💬",
    layout="wide",
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stTextArea textarea {font-size: 16px;}
        .result-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #e9f5ff;
            border-left: 6px solid #1f77b4;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title & Description ---
st.markdown("<h1 style='text-align:center; color:#1f77b4;'>🧠 NLP Text Classification App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>TF-IDF + Logistic Regression Model with 86% Accuracy</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Input Section ---
st.subheader("📌 Enter any text you want to classify:")
user_input = st.text_area("Your text here...", height=150, placeholder="e.g., I really enjoyed the service, very fast and polite!")

# --- Example Texts ---
with st.expander("📄 Try Examples"):
    st.markdown("""
    - *This product is absolutely amazing and exceeded my expectations.*
    - *Worst experience ever. Very disappointed.*
    - *The service was okay, not too great, not too bad.*
    """)

# --- Prediction Logic ---
if st.button("🔍 Predict Text Category"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        confidence = np.max(probabilities) * 100

        # --- Result Display ---
        st.markdown("### 🧾 Prediction Result")
        st.markdown(f"<div class='result-box'><b>Predicted Category:</b> {prediction}<br><b>Confidence:</b> {confidence:.2f}%</div>", unsafe_allow_html=True)

        # --- Confidence Chart ---
        st.markdown("### 📊 Prediction Probabilities")
        labels = model.classes_
        st.bar_chart({label: prob for label, prob in zip(labels, probabilities)})

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center; font-size:13px;'>Built with ❤️ by Sachin using Streamlit + Scikit-learn</p>", unsafe_allow_html=True)
