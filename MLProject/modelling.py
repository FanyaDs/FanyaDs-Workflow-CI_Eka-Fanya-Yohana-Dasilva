import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# === [1] Load dataset hasil preprocessing ===
df = pd.read_csv("MLProject/namadataset_preprocessing/wisata_bali_preprocessed.csv")

# Pastikan kolom sesuai dataset kamu
X = df["cleaned_text"]       # ganti sesuai nama kolom input
y = df["label"]              # ganti sesuai nama kolom label

# === [2] Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === [3] Vektorisasi sederhana (contoh dummy) ===
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === [4] Jalankan MLflow Tracking ===
mlflow.set_tracking_uri("file:///tmp/mlruns")  # lokal tracking
mlflow.set_experiment("Workflow_CI_Fanya")

with mlflow.start_run() as run:
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "model")

    print(f"Run ID: {run.info.run_id}")
    print(f"Akurasi model: {acc:.4f}")

print("Training selesai ✅")
