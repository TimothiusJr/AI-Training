import sys
import os
import numpy as np
import pandas as pd
import librosa
import joblib

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "svm_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))


FEATURE_COLUMNS = (
        [f"mfcc_mean_{i+1}" for i in range(13)] +
        [f"mfcc_std_{i+1}" for i in range(13)] +
        ["zcr", "rms", "spectral_centroid"]
)

LABEL_MAP = {0: "healthy", 1: "impaired"}

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
    rms = np.mean(librosa.feature.rms(y=y)[0])
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])

    return np.concatenate([mfccs_mean, mfccs_std, [zcr, rms, spectral_centroid]])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_wav>")
        sys.exit(1)

    wav_file = sys.argv[1]
    if not os.path.isfile(wav_file):
        print("❌ File not found:", wav_file)
        sys.exit(1)

    # Extract features
    features = extract_features(wav_file)
    feature_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    # Scale
    features_scaled = scaler.transform(feature_df)

    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    confidence = probabilities[prediction]

    print(
        f"🧠 Predicted: {LABEL_MAP[prediction]} "
        f"(confidence: {confidence:.2f})"
    )
