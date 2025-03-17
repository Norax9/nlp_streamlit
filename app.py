import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the best model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

def preprocess_text(texts, tokenizer, max_len=200):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    st.pyplot(plt)

# Streamlit UI
st.title("Fake News Detection App - Conv1D Model")

# Load model
model = load_model()

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Instruction text
st.write("### Enter a news article, and the model will predict whether it is real or fake.")

# Option to upload dataset or enter text manually
option = st.radio("Choose input method:", ("Upload CSV", "Enter Text"))

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload validation data (CSV)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Sample Data:")
        st.write(data.head())
        
        # Preprocess text
        texts = data["text"].astype(str).tolist()
        X_test = preprocess_text(texts, tokenizer)
        y_test = data["label"].values
        
        # Model Prediction
        y_pred_probs = model.predict(X_test)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader("Model Performance")
        st.write("Accuracy:", accuracy)
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, "Conv1D Model")

elif option == "Enter Text":
    user_text = st.text_area("Enter news text for prediction:")
    if st.button("Predict"): 
        if user_text.strip():
            X_input = preprocess_text([user_text], tokenizer)
            y_pred_prob = model.predict(X_input)[0, 0]
            y_pred_label = "Real" if y_pred_prob > 0.5 else "Fake"
            st.subheader("Prediction Result")
            st.write(f"Prediction: {y_pred_label} (Confidence: {y_pred_prob:.4f})")
        else:
            st.warning("Please enter some text to predict.")

