import streamlit as st
import joblib

# Page config
st.set_page_config(
    page_title="Flipkart Review Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# UI
st.title("🛒 Flipkart Review Sentiment Analyzer")
st.write("Enter a product review to predict sentiment")

review = st.text_area("Customer Review", height=150)

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        prediction = model.predict([review])[0]

        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")

st.markdown("---")
st.caption("Model: Naive Bayes | Trained on Flipkart Reviews")
