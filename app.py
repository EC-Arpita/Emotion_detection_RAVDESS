# app.py
import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
import librosa

model_path = "cnn_model.h5" 

# Load the CNN model
model = load_model(model_path)

st.title("🎭 Emotion Detection from Speech")
st.write("Upload an audio file to detect emotion.")

uploaded_file = st.file_uploader("Choose an audio file (wav, mp3)", type=["wav","mp3"])

if uploaded_file is not None:
    # Load audio file
    y, sr = librosa.load(uploaded_file, sr=16000)

    # -------------------------
    # Feature extraction (Log-Mel spectrogram for CNN)
    # -------------------------
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Fix length to 128 frames (same as training preprocessing)
    mel_spec_db = librosa.util.fix_length(mel_spec_db, size=128, axis=1)

    # Prepare input for CNN: (128, 128, 1)
    input_data = np.expand_dims(mel_spec_db, axis=-1)   # (128,128,1)
    input_data = np.expand_dims(input_data, axis=0)     # (1,128,128,1)

    # -------------------------
    # Prediction
    # -------------------------
    prediction = model.predict(input_data)
    class_index = np.argmax(prediction)

    # Emotion labels (adjust order according to your training)
    labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    predicted_emotion = labels[class_index]

    st.write("**Predicted Emotion:**", predicted_emotion)
    st.write("🔮 Model Confidence:", np.max(prediction).round(3))