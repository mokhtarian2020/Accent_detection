import os
import tempfile
from moviepy.editor import VideoFileClip
import whisper
import torchaudio
from speechbrain.pretrained.interfaces import foreign_class
import yt_dlp

# Load Whisper and Accent Classifier once
def load_whisper():
    return whisper.load_model("base")

def load_accent_model():
    return foreign_class(
        source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier"
    )

# ✅ Download video using yt_dlp (with correct extension)
def download_video(url):
    # If it's a local file, return it directly
    if os.path.isfile(url):
        print("📁 Local file detected, skipping download.")
        return url

    # Otherwise, download with yt_dlp
    temp_template = os.path.join(tempfile.gettempdir(), "temp_video.%(ext)s")
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': temp_template,
        'quiet': True,
        'noplaylist': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_file = ydl.prepare_filename(info)

    if not downloaded_file.endswith(".mp4"):
        raise ValueError("Downloaded file is not an MP4 video. Please use a valid .mp4 video.")

    return downloaded_file

# Extract and resample audio
def extract_audio(video_path):
    temp_audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
    waveform, sr = torchaudio.load(temp_audio_path)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    waveform = resampler(waveform)
    torchaudio.save(temp_audio_path, waveform, 16000)
    return temp_audio_path

# Transcribe
def transcribe_audio(audio_path, model):
    result = model.transcribe(audio_path)
    return result['text']

# Classify accent
def classify_accent(audio_path, classifier):
    out_prob, score, index, text_lab = classifier.classify_file(audio_path)
    return text_lab, round(score * 100, 2)

# Generate short explanation summary
def generate_summary(accent, confidence):
    summaries = {
        "US": "The speaker's accent closely aligns with standard American English, likely from the United States.",
        "EN": "The speech pattern reflects a British English accent, possibly from England or nearby regions.",
        "AU": "The accent resembles Australian English, characterized by flattened vowels and informal intonation.",
        "IN": "The accent is Indian English, with clear syllables and distinct stress patterns common in South Asia.",
        "CA": "This resembles Canadian English, often similar to American but with subtle vowel shifts.",
        "IE": "An Irish English accent, with musical intonation and non-rhotic patterns, is detected.",
    }
    return summaries.get(accent, "The detected accent indicates English speech with patterns that are less regionally specific or mixed.")

# MAIN CLI execution
def run_pipeline(video_url):
    print("⬇️  Downloading video...")
    video_path = download_video(video_url)

    print("🔊 Extracting and resampling audio...")
    audio_path = extract_audio(video_path)

    print("🧠 Loading Whisper...")
    whisper_model = load_whisper()

    print("📝 Transcribing audio...")
    transcript = transcribe_audio(audio_path, whisper_model)

    print("🌍 Loading accent classifier...")
    accent_model = load_accent_model()

    print("📊 Classifying accent...")
    accent, confidence = classify_accent(audio_path, accent_model)
    summary = generate_summary(accent, confidence)

    print("\n✅ RESULT:")
    print(f"Detected Accent     : {accent}")
    print(f"Confidence Score    : {confidence}%")
    print(f"Summary             : {summary}")
    print(f"Transcript (preview): {transcript[:300]}{'...' if len(transcript) > 300 else ''}")

if __name__ == "__main__":
    print("🎥 English Accent Detection CLI")
    video_url = input("Paste MP4 or YouTube video URL: ").strip()
    if video_url:
        run_pipeline(video_url)
    else:
        print("❌ No URL provided.")
