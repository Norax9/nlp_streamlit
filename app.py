import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from tensorflow.keras.models import load_model

# Download NLTK resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the pre-trained TF-IDF vectorizer and Keras model
tfidf = joblib.load('tfidf_vectorizer.pkl')
model = load_model('best_model.h5')

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords and apply lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    # Join back into a cleaned string
    return " ".join(tokens)

# Streamlit app
def main():
    st.title("Fake News Detector")
    st.write("Enter a news headline or article to check if it's real or fake.")

    # Input text box
    user_input = st.text_area("Input News Text", "")

    if st.button("Predict"):
        if user_input:
            # Preprocess the input text
            processed_text = preprocess_text(user_input)
            
            # Transform the text using the TF-IDF vectorizer
            text_tfidf = tfidf.transform([processed_text]).toarray()
            
            # Make a prediction
            prediction = model.predict(text_tfidf)
            
            # Display the result
            if prediction[0] > 0.5:  # Assuming binary classification with sigmoid output
                st.success("This news is **REAL**.")
            else:
                st.error("This news is **FAKE**.")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()