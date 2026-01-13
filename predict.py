import sys
import os
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl") if os.path.exists("scaler.pkl") else None

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
    rms = np.mean(librosa.feature.rms(y=y)[0])
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])

    return np.concatenate([mfccs_mean, mfccs_std, [zcr, rms, centroid]])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_wav>")
        sys.exit(1)

    wav_file = sys.argv[1]
    if not os.path.isfile(wav_file):
        print("❌ File not found:", wav_file)
        sys.exit(1)

    # Extract and scale features
    features = extract_features(wav_file).reshape(1, -1)
    if scaler:
        features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)[0]
    label_map = {0: "healthy", 1: "impaired"}
    print(f"🧠 Predicted: {label_map.get(prediction, 'unknown')}")

    
