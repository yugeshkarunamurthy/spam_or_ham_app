import streamlit as st
import joblib
import re

# Load the model
model = joblib.load('spam_classifier.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to predict the class of a new message
def predict_spam_ham(message):
    # Preprocess the message
    vectorized_message = loaded_vectorizer.transform([message])

    # Predict using the loaded model
    prediction = model.predict(vectorized_message)
    
    # Assuming your label_encoder mapped 'ham' to 0 and 'spam' to 1
    predicted_label = "spam" if prediction[0] == 1 else "ham"

    return predicted_label

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
        label = predict_spam_ham(user_input)
        st.success(f"Prediction: **{label}**")
