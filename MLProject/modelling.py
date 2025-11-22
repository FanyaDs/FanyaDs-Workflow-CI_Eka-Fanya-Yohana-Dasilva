import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import joblib

print("==============================================")
print("🚀 [DEBUG] modelling.py berhasil dijalankan")
print("📂 Working Directory :", os.getcwd())
print("==============================================")

# ===============================================================
# [1] Load Dataset
# ===============================================================
try:
    base_path = os.getcwd()
    data_path = os.path.join(base_path, "namadataset_preprocessing", "wisata_bali_preprocessed.csv")

    print(f"🔍 Mencoba memuat dataset dari: {data_path}")

    df = pd.read_csv(data_path)
    df = df.dropna(subset=["clean_text", "label"])
    df["clean_text"] = df["clean_text"].astype(str).str.strip()

    print(f"✅ Dataset berhasil dimuat ({len(df)} baris)")
except Exception as e:
    print(f"❌ Gagal memuat dataset: {e}")
    df = pd.DataFrame({"clean_text": ["fallback"], "label": [0]})

# ===============================================================
# [2] Train/Test Split
# ===============================================================
X = df["clean_text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================================================
# [3] Vectorizer
# ===============================================================
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===============================================================
# [4] Nested Run — FIX untuk MLflow Project
# ===============================================================
print("🧭 Memulai nested MLflow run...")

with mlflow.start_run(nested=True):
    print("🚀 Training model...")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("vectorizer", "CountVectorizer")

    mlflow.sklearn.log_model(clf, artifact_path="model")

    artifacts_dir = os.path.join(base_path, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    vec_path = os.path.join(artifacts_dir, "vectorizer.pkl")
    joblib.dump(vectorizer, vec_path)
    mlflow.log_artifact(vec_path)

    report_path = os.path.join(artifacts_dir, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact(report_path)

print("🎉 Training selesai tanpa error!")
print("==============================================")
