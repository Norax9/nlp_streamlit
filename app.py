import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

# Load the best model
def load_model():
    return tf.keras.models.load_model("best_model.h5")

# Preprocess text
def preprocess_text(texts, tokenizer, max_len=200):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

# Streamlit UI
st.title("Fake News Detection App - Conv1D Model")

# Load model
model = load_model()

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Instruction text
st.write("### Choose an option to predict: Upload a CSV file or enter a news article.")

# Option to upload dataset or enter text manually
option = st.radio("Choose input method:", ("Upload CSV", "Enter Text"))

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload validation data (CSV)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Sample Data:")
        st.write(data.head())
        
        # Ensure 'text' and 'label' columns exist
        if "text" not in data.columns or "label" not in data.columns:
            st.error("CSV must contain 'text' and 'label' columns.")
        else:
            # Preprocess text
            texts = data["text"].astype(str).tolist()
            X_test = preprocess_text(texts, tokenizer)
            y_test = data["label"].to_numpy()

            # Debug: Print unique values in y_test
            st.write("Unique values in y_test before fixing:", np.unique(y_test))

            # Map labels: 2 -> 1 (Real), others -> 0 (Fake)
            y_test = np.where(y_test == 2, 1, 0)
            st.write("Final unique values in y_test after replacement:", np.unique(y_test))

            # Model Prediction
            y_pred_probs = model.predict(X_test)
            y_pred = (y_pred_probs.squeeze() > 0.5).astype(int)

            # Add predictions to the dataframe
            data["Prediction"] = ["Real" if pred == 1 else "Fake" for pred in y_pred]
            st.write("Predictions:")
            st.write(data[["text", "Prediction"]])

            # Evaluate model
            accuracy = accuracy_score(y_test, y_pred)
            st.subheader("Model Performance")
            st.write("Accuracy:", accuracy)
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

elif option == "Enter Text":
    st.write("### Enter a news article to predict if it's Fake or Real.")
    user_input = st.text_area("Paste the news article here:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.error("Please enter some text to predict.")
        else:
            # Preprocess the input text
            X_input = preprocess_text([user_input], tokenizer)

            # Predict
            y_pred_prob = model.predict(X_input)
            y_pred = (y_pred_prob.squeeze() > 0.5).astype(int)

            # Display prediction
            st.subheader("Prediction Result")
            if y_pred[0] == 0:
                st.error("Prediction: Fake News")
            else:
                st.success("Prediction: Real News")

            # Show prediction probability
            st.write(f"Prediction Probability: {y_pred_prob.squeeze():.4f}")
