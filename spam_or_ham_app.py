import streamlit as st
import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load the model
model = joblib.load('spam_classifier.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def Clean(Text):
    sms = re.sub('[^a-zA-Z]', ' ', Text)
    sms = sms.lower()
    sms = sms.split()
    sms = ' '.join(sms)
    return sms

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in text if word not in stop_words]
    return filtered_text

def lemmatize_word(text):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in text]
    return lemmas

# Function to predict the class of a new message
def predict_spam_ham(message):
    # Preprocess the message
    cleaned_text = Clean(message)
    tokenized_text = nltk.word_tokenize(cleaned_text)
    nostopword_text = remove_stopwords(tokenized_text)
    lemmatized_text = lemmatize_word(nostopword_text)
    preprocessed_message = ' '.join(lemmatized_text)

    # Vectorize the preprocessed message using the loaded vectorizer
    vectorized_message = loaded_vectorizer.transform([preprocessed_message])

    # Predict using the loaded model
    prediction = model.predict(vectorized_message)

    # Decode the prediction (0 for ham, 1 for spam - based on how you encoded it)
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
