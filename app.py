import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os
import re
from PIL import Image

st.set_page_config(page_title="Amazon Review Analysis", page_icon="🛡️", layout="wide")

# Custom CSS for that "Eye-catching" and "Classy" look
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------------------------------------
# LOAD MODELS & SCALERS
# -----------------------------------------------------
@st.cache_resource
def load_models():
    try:
        with open("random_forest.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("shap_explainer.pkl", "rb") as f:
            explainer = pickle.load(f)
        
        # Load sample training data to calculate means
        X_train = pd.read_csv("X_train_processed.csv")
        return model, vectorizer, explainer, X_train
    except Exception as e:
        st.error(f"Error loading models: {e}. Please run the training scripts first.")
        st.stop()
        return None, None, None, None

model, vectorizer, explainer, X_train_sample = load_models()

# -----------------------------------------------------
# UI HEADER
# -----------------------------------------------------
st.markdown("<h1>🛡️ Fake Review Detection Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Powered by Natural Language Processing and Behavioral Analytics</p>", unsafe_allow_html=True)

# -----------------------------------------------------
# INPUT SECTION
# -----------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Analyze a Review")
    review_text = st.text_area("Paste the review text here:", height=150, 
                               placeholder="e.g., BEST PRODUCT EVER!!! BOUGHT 10 OF THEM. HIGHLY RECOMMEND!!")
    
with col2:
    st.subheader("👤 Reviewer Context")
    rating = st.slider("Product Rating Given (1-5 Stars):", 1, 5, 5)
    verified = st.radio("Is this a Verified Purchase?", ["Yes", "No"], index=1)
    # Simulate a product average to calculate deviation
    product_avg = st.slider("Average Rating of this Product:", 1.0, 5.0, 4.2, 0.1)
    user_total_reviews = st.number_input("How many reviews has this user left?", min_value=1, max_value=5000, value=1)

analyze_button = st.button("🔍 Run Fraud Analysis")

st.markdown("---")

# -----------------------------------------------------
# INFERENCE LOGIC
# -----------------------------------------------------
if analyze_button and review_text.strip():
    with st.spinner("Running deep behavioral analysis and NLP vectorization..."):
        
        # 1. Engineer Real-Time Features
        review_length = len(review_text.split())
        exclamation_count = review_text.count('!')
        is_verified = 1 if verified == "Yes" else 0
        rating_deviation = abs(rating - product_avg)
        
        # 2. Vectorize the text
        tfidf_features = vectorizer.transform([review_text]).toarray()
        
        # 3. Combine into a single DataFrame that matches training EXACTLY
        # The training columns were: [length, exclamation, verified, review_count, deviation] + [500 tfidf words]
        numeric_features = [review_length, exclamation_count, is_verified, user_total_reviews, rating_deviation]
        
        # Build feature names
        num_cols = ['review_length', 'exclamation_count', 'verified_purchase', 'reviewer_review_count', 'rating_deviation']
        tfidf_cols = vectorizer.get_feature_names_out()
        all_cols = num_cols + list(tfidf_cols)
        
        # Create final input row
        input_row = np.hstack([numeric_features, tfidf_features[0]])
        X_input = pd.DataFrame([input_row], columns=all_cols)
        
        # 4. Predict
        probability = model.predict_proba(X_input)[0][1] # Probability class 1 (Fake)
        prediction = 1 if probability > 0.5 else 0
        
        # -----------------------------------------------------
        # RESULTS DASHBOARD
        # -----------------------------------------------------
        st.subheader("🎯 Analysis Results")
        
        # Score banner
        if prediction == 1:
            st.markdown(f"<div class='fake-box'>⚠️ SUSPICIOUS ({probability*100:.1f}% Probability of being Fake)</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='genuine-box'>✅ GENUINE ({(1-probability)*100:.1f}% Probability of being Authentic)</div>", unsafe_allow_html=True)
            
        st.write("")
        
        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"<div class='metric-container'><div class='metric-value'>{review_length}</div><div class='metric-label'>Word Count</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-container'><div class='metric-value'>{exclamation_count}</div><div class='metric-label'>Exclamations</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-container'><div class='metric-value'>{rating_deviation:.1f}</div><div class='metric-label'>Rating Dev.</div></div>", unsafe_allow_html=True)
        with m4:
            st.markdown(f"<div class='metric-container'><div class='metric-value'>{user_total_reviews}</div><div class='metric-label'>User History</div></div>", unsafe_allow_html=True)
            
        st.write("")
        st.write("")
        
        # -----------------------------------------------------
        # SHAP EXPLAINABILITY
        # -----------------------------------------------------
        st.subheader("🧠 Why was this decision made? (SHAP Analysis)")
        st.write("The chart below shows exactly which behavioral signals or semantic words pushed the model toward predicting 'Fake' (Red) or 'Genuine' (Blue).")
        
        try:
            # Calculate SHAP values for this specific review
            shap_values = explainer(X_input)
            
            # Create a localized matplotlib figure for Streamlit
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Show the waterfall plot
            # shap outputs explainers differently based on model type. For RF it's an array of explainers per class
            # We want class 1 (Fake)
            shap.plots.waterfall(shap_values[0, :, 1], max_display=10, show=False)
            
            # Clean up the plot aesthetic
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation chart: {e}")
            
elif analyze_button and not review_text.strip():
    st.error("Please enter a review text to analyze.")
