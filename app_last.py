import yt_dlp
import os
import torchaudio
import torch
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import fetch  # for file-safe copying on Windows
import shutil

DEBUG = True
TEMP_AUDIO = "temp_audio.wav"

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
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return TEMP_AUDIO
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

import shutil

def classify_accent(audio_path):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        source_dir = "./accent-id-commonaccent_ecapa"
        target_dir = "pretrained_models/accent-id-commonaccent_ecapa"
        os.makedirs(target_dir, exist_ok=True)

        # ✅ Manually copy required files
        required_files = ["hyperparams.yaml", "model.ckpt", "label_encoder.txt", "valid.csv"]
        for filename in required_files:
            src_file = os.path.join(source_dir, filename)
            dst_file = os.path.join(target_dir, filename)
            if not os.path.exists(dst_file):  # Avoid overwriting every run
                shutil.copyfile(src_file, dst_file)

        # Load classifier from local target
        classifier = EncoderClassifier.from_hparams(
            source=target_dir,
            run_opts={"device": device}
        )

        out_prob, score, index, text_lab = classifier.classify_file(audio_path)
        print(f"Model is running on device: {device}")
        return text_lab[0], float(score[0])

    except Exception as e:
        raise Exception(f"Classification failed: {str(e)}")


def test():
    print("\n=== Starting Test ===")
    url = "https://www.youtube.com/watch?v=X627czLUsGY"

    try:
        print("Downloading audio...")
        audio_path = download_audio(url)

        print("Classifying accent...")
        accent, confidence = classify_accent(audio_path)

        print("\n=== Results ===")
        print(f"Detected Accent: {accent}")
        print(f"Confidence: {confidence * 100:.1f}%")

    except Exception as e:
        print(f"\n!!! Error: {str(e)}")
    finally:
        if os.path.exists(TEMP_AUDIO):
            os.remove(TEMP_AUDIO)
            if DEBUG:
                print(f"Deleted temporary file: {TEMP_AUDIO}")

if __name__ == "__main__":
    test()
