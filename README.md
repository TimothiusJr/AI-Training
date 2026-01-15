 # AI Speech Classification System

This project is an end-to-end machine learning system that analyzes speech recordings to detect patterns associated with neurodegenerative impairment. It combines a Python-based ML pipeline with a Node.js REST API and SQL persistence to deliver confidence-aware predictions and store inference history.

---

## Overview

Speech is often affected early in neurodegenerative conditions through changes in articulation, energy, and frequency patterns. This project explores speech-based detection as a non-invasive screening approach using classical machine learning and audio signal processing.

The system is designed as a small, production-style service rather than a standalone script.

---

## Architecture

Client  
↓  
Node.js (Express API)  
↓  
Python (ML Inference)  
↓  
SQLite (Prediction Logging)

- Python handles feature extraction and machine learning inference  
- Node.js (Express) exposes the model through REST endpoints  
- SQLite stores prediction results and confidence scores  

---

## Machine Learning Pipeline

### Feature Extraction

Speech audio is processed using Librosa to extract acoustic features:

- MFCC means (13)
- MFCC standard deviations (13)
- Zero-Crossing Rate (ZCR)
- Root Mean Square energy (RMS)
- Spectral centroid

### Model

- Algorithm: Support Vector Machine (SVM)
- Kernel: RBF
- Feature scaling: StandardScaler
- Training: Supervised binary classification (healthy vs impaired)
- Output: Predicted label with confidence score

### Performance

- Accuracy: 92%
- Healthy recall: 100%
- Impaired recall: 80%

---

## Confidence Scores

Each prediction includes a confidence score derived from class probabilities.

Example:
Predicted: healthy (confidence: 0.72)

The confidence score reflects how strongly the input matches learned speech patterns.

---

## API Endpoints

### POST /predict

- Accepts a WAV audio file
- Runs ML inference using the Python model
- Logs the prediction and confidence score to SQL
- Returns structured JSON output

Example response:
{
  "label": "healthy",
  "confidence": 0.72
}

### GET /predictions

- Returns recent prediction history from the SQL database

Example response:
[
  {
    "id": 1,
    "filename": "healthy_01.wav",
    "label": "healthy",
    "confidence": 0.72,
    "created_at": "2026-01-15 07:15:42"
  }
]

---

## Database

- Database: SQLite
- Purpose: Persist prediction results
- Stored fields:
  - filename
  - predicted label
  - confidence score
  - timestamp

---

## Project Structure

AI-Speech-Classification/  
├── api/  
│   ├── server.js  
│   ├── package.json  
│   ├── package-lock.json  
│   └── uploads/ (ignored)  
├── audio/  
├── build_dataset.py  
├── extract_features.py  
├── train_model.py  
├── predict.py  
├── README.md  
└── .gitignore  

---

## How to Run

Python setup:
pip install librosa numpy pandas scikit-learn joblib soundfile

Train the model:
python3 build_dataset.py  
python3 train_model.py  

Start the API:
cd api  
npm install  
node server.js  

Make a prediction:
curl -X POST -F "audio=@audio/healthy_01.wav" http://localhost:3000/predict

View prediction history:
curl http://localhost:3000/predictions

---

## Technologies Used

- Python
- scikit-learn
- librosa
- Node.js
- Express.js
- SQLite
- REST APIs
- Git / GitHub

---

## Limitations

- Small dataset size
- Performance depends on audio quality
- Intended for research and learning, not medical diagnosis

---

## Author

Tee-jay Salmon
