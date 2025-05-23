# 🎙️ English Accent Detection (Streamlit Demo)

This is a Streamlit web application that identifies the **English accent** of a speaker from a ** video link**.

It uses the [SpeechBrain](https://speechbrain.readthedocs.io) library and a pre-trained **ECAPA-TDNN model** fine-tuned on the [CommonAccent dataset](https://huggingface.co/Jzuluaga/accent-id-commonaccent_ecapa). The system classifies **16 English accents** and provides a confidence score.

---

##Structure overview: 
├── streamlit_app.py                # Main Streamlit interface
├── requirements.txt                # Python dependencies
├── pretrained_models/
│   └── accent-id-commonaccent_ecapa/
│       ├── model.ckpt
│       ├── hyperparams.yaml
│       ├── label_encoder.txt
│       └── valid.csv



## 🔍 Features

- ✅ Paste any **Video link** with spoken English
- ✅ The app downloads and extracts the speech
- ✅ Detects and classifies the **speaker's English accent**
- ✅ Outputs:
  - Accent label (e.g., *British*, *American*)
  - Confidence score (0–100%)
  - Natural language summary



## 🚀 Run Locally (GPU or CPU)

1. Clone the repository:
    git clone https://github.com/mokhtarian2020/Accent_detection.git
    cd Accent_detection

2. Install dependencies (Python 3.10+ required):
    pip install -r requirements.txt

   Also install required system packages:
    sudo apt-get install ffmpeg libsndfile1

3. Run the app:
    streamlit run streamlit_app.py

Then open your browser at: http://localhost:8501

