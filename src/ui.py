
import streamlit as st
import requests
import pandas as pd
import json

# Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Customer Feedback Sentiment AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Customer Feedback Sentiment AI")
st.markdown("Analyze customer feedback sentiment using our advanced ML model.")

# Health Check
try:
    health = requests.get(f"{API_URL}/")
    if health.status_code == 200:
        st.success(f"System Online (v{health.json().get('version')})")
    else:
        st.error("System Unhealthy")
except requests.exceptions.ConnectionError:
    st.error("⚠️ API is not running. Please start the backend server.")
    st.code("python -m uvicorn api.main:app --reload")

# Tabs
tab1, tab2 = st.tabs(["Single Prediction", "Batch Analysis"])

with tab1:
    st.header("Analyze Feedback")
    text_input = st.text_area("Enter customer review:", height=150, placeholder="e.g., The product quality was amazing, but shipping was slow.")
    
    if st.button("Analyze Sentiment"):
        if text_input.strip():
            try:
                with st.spinner("Analyzing..."):
                    payload = {"text": text_input}
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        sentiment = result['sentiment']
                        confidence = result['confidence']
                        
                        # Display result
                        col1, col2 = st.columns(2)
                        with col1:
                            if sentiment == "Positive":
                                st.success(f"**{sentiment}**")
                            elif sentiment == "Negative":
                                st.error(f"**{sentiment}**")
                            else:
                                st.warning(f"**{sentiment}**")
                        
                        with col2:
                            st.metric("Confidence Score", f"{confidence:.1%}")
                            st.progress(confidence)
                            
                    else:
                        st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
        else:
            st.warning("Please enter some text.")

with tab2:
    st.header("Batch Analysis")
    uploaded_file = st.file_uploader("Upload CSV (must contain 'text' column)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns:
            if st.button("Analyze Batch"):
                with st.spinner(f"Analyzing {len(df)} reviews..."):
                    texts = df['text'].astype(str).tolist()
                    # Batch in chunks of 50 to avoid timeouts/limits if needed
                    # For now, one batch
                    payload = {"texts": texts}
                    
                    try:
                        response = requests.post(f"{API_URL}/batch_predict", json=payload)
                        if response.status_code == 200:
                            results = response.json()['predictions']
                            
                            # Add results to dataframe
                            df['predicted_sentiment'] = [r['sentiment'] for r in results]
                            df['confidence'] = [r['confidence'] for r in results]
                            
                            st.dataframe(df)
                            
                            # Analysis charts
                            st.subheader("Distribution")
                            st.bar_chart(df['predicted_sentiment'].value_counts())
                            
                            # Download
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Results",
                                csv,
                                "sentiment_results.csv",
                                "text/csv",
                                key='download-csv'
                            )
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
        else:
            st.error("CSV must contain a 'text' column.")
