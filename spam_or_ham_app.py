import streamlit as st
import joblib

# Load the model
model = joblib.load('spam_classifier.pkl')

# App title
st.title("📩 Spam or Ham Classifier")
st.write("Enter a message to check if it's spam or ham.")

# Input text
user_input = st.text_area("Message", placeholder="Enter your message here...")

# Predict
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([user_input])[0]
        label = "🚫 Spam" if prediction == 1 else "✅ Ham"
        st.success(f"Prediction: **{label}**")
