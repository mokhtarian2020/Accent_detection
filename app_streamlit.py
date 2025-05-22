import streamlit as st
import yt_dlp
import subprocess
import librosa
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Mapping model's output labels to user-friendly names
label_mapping = {
    'us': 'American',
    'england': 'British',
    'australia': 'Australian',
}

def download_video(url):
    """Download video from URL using yt-dlp."""
    ydl_opts = {
        'outtmpl': 'temp_video',
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info_dict)
    return filename

def extract_audio(video_path, audio_path='temp_audio.wav'):
    """Extract audio from video using FFmpeg."""
    command = f"ffmpeg -i {video_path} -ab 160k -ac 1 -ar 16000 -vn {audio_path}"
    subprocess.run(command, shell=True, check=True)
    return audio_path

def classify_accent(audio_path):
    """Classify accent using a pre-trained model."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    # Initialize model and processor
    model_name = "sahilkhosla/accent-classification-commonvoice"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
    # Process audio
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Get predictions
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    confidence, predicted_idx = torch.max(probabilities, dim=0)
    predicted_label = model.config.id2label[predicted_idx.item()]
    # Map label to user-friendly name
    return label_mapping.get(predicted_label, predicted_label), confidence.item()

# Streamlit UI
st.title("Accent Detection Tool")
url = st.text_input("Enter Video URL")

if url:
    try:
        video_path = download_video(url)
        audio_path = extract_audio(video_path)
        accent, confidence = classify_accent(audio_path)
        st.success(f"**Accent:** {accent}")
        st.success(f"**Confidence:** {confidence * 100:.2f}%")
    except Exception as e:
        st.error(f"Error processing video: {e}")