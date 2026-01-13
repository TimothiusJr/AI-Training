import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc, sr

def plot_mfcc(mfcc, sr, file_name):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title(f'MFCC of {file_name}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    audio_file = "sample.wav"  # Make sure this file exists in your project folder
    if os.path.exists(audio_file):
        mfcc, sr = extract_mfcc(audio_file)
        plot_mfcc(mfcc, sr, os.path.basename(audio_file))
    else:
        print("Audio file not found. Please check your file path.")
