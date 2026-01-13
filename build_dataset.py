import os
import librosa
import numpy as np
import pandas as pd

# Folder with audio files
AUDIO_DIR = "audio"  # make sure this folder exists
OUTPUT_CSV = "features.csv"

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
    rms = np.mean(librosa.feature.rms(y=y)[0])
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])

    # Combine all features into a flat array
    features = np.concatenate([mfccs_mean, mfccs_std, [zcr, rms, spectral_centroid]])
    return features

# Labeling strategy based on file name
def get_label(file_name):
    if "healthy" in file_name.lower():
        return "healthy"
    elif "impaired" in file_name.lower():
        return "impaired"
    else:
        return "unknown"

# Process all files
def process_directory(audio_dir):
    feature_list = []
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(audio_dir, file_name)
            try:
                features = extract_features(file_path)
                label = get_label(file_name)
                feature_list.append([file_name] + features.tolist() + [label])
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Column headers
    columns = ["filename"]
    columns += [f"mfcc_mean_{i+1}" for i in range(13)]
    columns += [f"mfcc_std_{i+1}" for i in range(13)]
    columns += ["zcr", "rms", "spectral_centroid", "label"]

    # Save to CSV
    df = pd.DataFrame(feature_list, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Feature dataset saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    process_directory(AUDIO_DIR)
