import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the best model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

def preprocess_text(texts, tokenizer, max_len=100):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    st.pyplot(plt)

# Streamlit UI
st.title("Fake News Detection App - Deep Learning Model")

# Upload dataset
uploaded_file = st.file_uploader("Upload validation data (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Sample Data:")
    st.write(data.head())
    
    # Load model
    model = load_model()
    
    # Load tokenizer
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(open("tokenizer.json").read())
    
    # Preprocess text
    X_test = preprocess_text(data["text"], tokenizer)
    y_test = data["label"]
    
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
    plot_confusion_matrix(y_test, y_pred, "Deep Learning Model")
