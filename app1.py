import streamlit as st
import yt_dlp
import subprocess
import librosa
import torch
import os
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# ====== Configuration ======
DEBUG = True  # Set to False for Streamlit deployment
TEMP_VIDEO = "temp_video.mp4"
TEMP_AUDIO = "temp_audio.wav"

# Mapping model's output labels to user-friendly names
LABEL_MAPPING = {
    'us': 'American',
    'england': 'British',
    'australia': 'Australian',
    'canada': 'Canadian',
    'indian': 'Indian',
    'other': 'Other'
}

# ====== Core Functions ======
def download_video(url: str) -> str:
    """More resilient video downloader"""
    ydl_opts = {
        'outtmpl': 'temp_video',
        'format': 'bestaudio/best',  # More flexible format selection
        'quiet': not DEBUG,
        'extract_audio': True,  # Directly extract audio
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '160',
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return f"temp_video.wav"  # yt-dlp will create this directly
    except Exception as e:
        raise Exception(f"Download failed. Try another URL. Error: {str(e)}")

def extract_audio(video_path: str, audio_path: str = TEMP_AUDIO) -> str:
    """Extract audio from video using FFmpeg."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    command = [
        "ffmpeg",
        "-i", video_path,
        "-ab", "160k",
        "-ac", "1",
        "-ar", "16000",
        "-vn", audio_path,
        "-y"  # Overwrite without asking
    ]
    
    try:
        if DEBUG:
            print(f"Extracting audio with command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=not DEBUG)
        return audio_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg failed: {e.stderr.decode('utf-8')}")

def classify_accent(audio_path: str) -> tuple:
    """Classify accent using a pre-trained model."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
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
        return LABEL_MAPPING.get(predicted_label, predicted_label), confidence.item()
    
    except Exception as e:
        raise Exception(f"Accent classification failed: {str(e)}")

# ====== Testing in VSCode ======
def test_locally(url: str):
    """Test the pipeline without Streamlit."""
    print("\n=== Starting Local Test ===")
    try:
        # Step 1: Download
        print(f"Downloading video from: {url}")
        video_path = download_video(url)
        
        # Step 2: Extract Audio
        print("Extracting audio...")
        audio_path = extract_audio(video_path)
        
        # Step 3: Classify
        print("Classifying accent...")
        accent, confidence = classify_accent(audio_path)
        
        # Results
        print("\n=== Results ===")
        print(f"Accent: {accent}")
        print(f"Confidence: {confidence * 100:.2f}%")
        
    except Exception as e:
        print(f"\n!!! Error: {str(e)}")
    finally:
        # Cleanup
        for f in [TEMP_VIDEO, TEMP_AUDIO]:
            if os.path.exists(f):
                os.remove(f)
                if DEBUG:
                    print(f"Deleted temporary file: {f}")

# ====== Streamlit UI ======
def streamlit_app():
    st.title("Accent Detection Tool")
    url = st.text_input("Enter Video URL (YouTube, Loom, etc.)")
    
    if st.button("Analyze") and url:
        with st.spinner("Processing..."):
            try:
                video_path = download_video(url)
                audio_path = extract_audio(video_path)
                accent, confidence = classify_accent(audio_path)
                
                st.success(f"**Accent:** {accent}")
                st.success(f"**Confidence:** {confidence * 100:.2f}%")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                # Cleanup
                for f in [video_path, TEMP_AUDIO]:
                    if f and os.path.exists(f):
                        os.remove(f)

# ====== Execution Control ======
if __name__ == "__main__":
    if DEBUG:
        # Test with a sample YouTube URL in VSCode
        test_locally("https://www.youtube.com/watch?v=dQw4w9WgXcQ")  # Replace with your test URL
    else:
        # Run Streamlit app
        streamlit_app()