import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("features.csv")


# Drop filename column
X = df.drop(columns=["filename", "label"])
y = df["label"]

# Encode labels (healthy = 0, impaired = 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.33, stratify=y_encoded, random_state=42
)


# Train SVM
clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

import joblib
joblib.dump(clf, "svm_model.pkl")
print("✅ Model saved as svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

