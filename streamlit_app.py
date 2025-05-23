import streamlit as st
import os
import shutil
import yt_dlp
import torch
import librosa
import numpy as np
from speechbrain.inference.classifiers import EncoderClassifier

DEBUG = True
TEMP_AUDIO = "temp_audio.wav"
SOURCE_DIR = "./accent-id-commonaccent_ecapa"
TARGET_DIR = "pretrained_models/accent-id-commonaccent_ecapa"

LABEL_MAP = {
    "us": "American",
    "england": "British",
    "australia": "Australian",
    "canada": "Canadian",
    "indian": "Indian",
    "scotland": "Scottish",
    "ireland": "Irish",
    "philippines": "Philippine",
    "wales": "Welsh",
    "african": "African English",
    "newzealand": "New Zealander",
    "hongkong": "Hong Kong English",
    "malaysia": "Malaysian English",
    "singapore": "Singaporean English",
    "bermuda": "Bermudian",
    "southatlandtic": "South Atlantic"
}

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '160',
        }],
        'outtmpl': TEMP_AUDIO.replace('.wav', ''),
        'quiet': not DEBUG,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return TEMP_AUDIO

def setup_model():
    os.makedirs(TARGET_DIR, exist_ok=True)
    for fname in ["hyperparams.yaml", "model.ckpt", "label_encoder.txt", "valid.csv"]:
        src = os.path.join(SOURCE_DIR, fname)
        dst = os.path.join(TARGET_DIR, fname)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source=TARGET_DIR, run_opts={"device": device})
    return classifier, device

def classify_accent(audio_path, classifier, device):
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    signal = torch.tensor(audio).unsqueeze(0).to(device)
    out_prob, score, index, text_lab = classifier.classify_batch(signal)
    return text_lab[0], float(score[0])

# === Streamlit UI ===
st.set_page_config(page_title="English Accent Classifier", page_icon="🎙️")
st.title("🎙️ English Accent Classifier (Demo)")
st.markdown("This demo identifies the **English accent** of a speaker from a video. Paste a link with clear English speech.")

url = st.text_input("🔗 Enter Video URL:")
if st.button("Analyze"):
    if not url.strip():
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Processing audio and loading model..."):
            try:
                audio_path = download_audio(url)
                classifier, device = setup_model()
                accent_raw, confidence = classify_accent(audio_path, classifier, device)
                friendly_accent = LABEL_MAP.get(accent_raw, accent_raw)

                st.success("✅ Accent classification complete!")
                st.markdown(f"**Accent:** {friendly_accent}")
                st.markdown(f"**Confidence:** {confidence * 100:.1f}%")
                st.markdown(f"**Summary:** The speaker in the video most likely has a **{friendly_accent}** English accent "
                            f"with a confidence score of **{confidence * 100:.1f}%**.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            finally:
                if os.path.exists(TEMP_AUDIO):
                    os.remove(TEMP_AUDIO)
