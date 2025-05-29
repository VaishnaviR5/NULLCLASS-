import streamlit as st
import numpy as np
import sounddevice as sd
import librosa
import tempfile
import os
import scipy.io.wavfile as wavfile
from keras.models import load_model

# Load model once
@st.cache_resource
def load_ser_model():
    return load_model(r'c:\Users\USER\Desktop\Speech Recognition Model\ser_model.h5')

model = load_ser_model()

# Define emotion labels (must match model training order)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

st.title("🎙️ Speech Emotion Recognition")
st.markdown("Upload or record your voice and let the model predict the emotion.")

# Audio recording parameters
DURATION = 4  # seconds
SAMPLE_RATE = 22050

# --- Extract MFCC features ---
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = mfcc.T
    if mfcc.shape[0] < 20:
        pad_width = 20 - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    mfcc = mfcc[:20]  # Take only the first 20 frames
    mfcc = np.mean(mfcc, axis=0)  # Average across time frames to get shape (20,)
    return mfcc

# --- Record audio from microphone ---
def record_audio(duration=DURATION):
    st.info("Recording... Speak now!")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

# --- Save recorded audio as WAV file ---
def save_audio_to_wav(audio, sr=SAMPLE_RATE):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wavfile.write(temp_file.name, sr, (audio * 32767).astype(np.int16))
    return temp_file.name

# --- Predict emotion from audio file ---
def predict_emotion(file_path):
    features = extract_mfcc(file_path)  # Shape: (20,)
    features = features.reshape(1, 20, 1)  # CNN expects (batch, time, features)
    prediction = model.predict(features)[0]
    predicted_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_index]
    return predicted_emotion

# === Upload option ===
st.subheader("📤 Upload your voice")
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file:
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path.write(uploaded_file.read())
    temp_path.close()

    st.audio(temp_path.name, format='audio/wav')

    with st.spinner("Analyzing..."):
        emotion = predict_emotion(temp_path.name)
        st.success(f"Predicted Emotion: **{emotion}**")

# === Record option ===
st.subheader("🎙️ Or record your voice")
if st.button("Start Recording"):
    audio = record_audio()
    temp_wav = save_audio_to_wav(audio)
    st.audio(temp_wav, format='audio/wav')

    with st.spinner("Analyzing..."):
        emotion = predict_emotion(temp_wav)
        st.success(f"Predicted Emotion: **{emotion}**")
