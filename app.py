import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('models/spam_classifier.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Streamlit app
st.title("📧 Spam Message Detector")
st.markdown("Enter a message and find out if it's **Spam or Ham**.")

# Input from user
user_input = st.text_area("Type your message here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Clean and vectorize
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        
        if prediction == 1:
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **VALID ONE**")
