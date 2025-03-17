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
def load_model():
    return tf.keras.models.load_model("best_model.h5")

# Preprocess text
def preprocess_text(texts, tokenizer, max_len=200):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

# Plot confusion matrix
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

            # Ensure labels are binary (0 or 1)
            y_test = np.array([0 if y == 2 else y for y in y_test])
            st.write("Final unique values in y_test after replacement:", np.unique(y_test))

            # Model Prediction
            y_pred_probs = model.predict(X_test)
            st.write("Sample y_pred_probs:", y_pred_probs[:10])  # Debugging output

            y_pred = (y_pred_probs.squeeze() > 0.5).astype(int)

            # Evaluate model
            accuracy = accuracy_score(y_test, y_pred)
            st.subheader("Model Performance")
            st.write("Accuracy:", accuracy)
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Plot confusion matrix
            plot_confusion_matrix(y_test, y_pred, "Conv1D Model")

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
