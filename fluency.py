import io
import librosa
import numpy as np
import pydub
from urllib.request import urlopen

# Load Model
def feature_extraction(file_name):
    wav = io.BytesIO()

    with urlopen(file_name) as r:
        r.seek = lambda *args: None  # allow pydub to call seek(0)
        pydub.AudioSegment.from_file(r).export(wav, "wav")

    wav.seek(0)
    X , sample_rate = librosa.load(wav)

    if X.ndim > 1:
        X = X[:,0]
    X = X.T
            
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=30).T, axis=0)
    rmse = np.mean(librosa.feature.rms(y=X).T, axis=0)
    spectral_flux = np.mean(librosa.onset.onset_strength(y=X, sr=sample_rate).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)

    return mfccs, rmse, spectral_flux, zcr