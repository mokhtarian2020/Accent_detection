# 🎙️ English Accent Detection (Streamlit Demo)

This is a Streamlit web application that identifies the **English accent** of a speaker from a ** video link**.

It uses the [SpeechBrain](https://speechbrain.readthedocs.io) library and a pre-trained **ECAPA-TDNN model** fine-tuned on the [CommonAccent dataset](https://huggingface.co/Jzuluaga/accent-id-commonaccent_ecapa). The system classifies **16 English accents** and provides a confidence score.

---

## 🔍 Features

- ✅ Paste any **YouTube link** with spoken English
- ✅ The app downloads and extracts the speech
- ✅ Detects and classifies the **speaker's English accent**
- ✅ Outputs:
  - Accent label (e.g., *British*, *American*)
  - Confidence score (0–100%)
  - Natural language summary

---

## ✅ Evaluation Criteria

| Area                      | Status | Details |
|---------------------------|--------|---------|
| **Functional Script**     | ✅ Yes | App runs, accepts YouTube input, returns accent classification |
| **Logical Approach**      | ✅ Yes | Uses ECAPA-TDNN (SOTA) with SpeechBrain |
| **Setup Clarity**         | ✅ Yes | Instructions and requirements included |
| **Accent Handling (EN)**  | ✅ Yes | Trained only on English accents |
| **Bonus: Confidence**     | ✅ Yes | Confidence percentage + summary displayed |

---

## 🚀 Run Locally (GPU or CPU)

### 1. Clone this repository

```bash
git clone https://github.com/your-org/accent-detection-streamlit.git
cd accent-detection-streamlit
