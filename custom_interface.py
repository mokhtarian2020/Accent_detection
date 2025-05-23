import torch
import torchaudio
from speechbrain.inference.interfaces import Pretrained

class CustomEncoderWav2vec2Classifier(Pretrained):
    def classify_file(self, path):
        signal, fs = torchaudio.load(path)

        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)

        # convert stereo to mono if needed
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        return self.classify_batch(signal)
