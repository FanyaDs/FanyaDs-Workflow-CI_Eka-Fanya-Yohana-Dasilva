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

# Gunakan autolog — MLflow yang membuka & menutup run otomatis
mlflow.autolog()

# ===============================================================
# [1] LOAD DATASET
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
# [2] SPLIT DATA
# ===============================================================
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================================================
# [3] VEKTORISASI
# ===============================================================
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===============================================================
# [4] TRAINING
# ===============================================================
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test_vec)

# METRICS MANUAL
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("==============================================")
print(f"🔢 Accuracy : {acc:.4f}")
print(f"🎯 Precision: {prec:.4f}")
print(f"📈 Recall   : {rec:.4f}")
print(f"🏆 F1-score : {f1:.4f}")
print("==============================================")

# ===============================================================
# [5] SIMPAN ARTEFAK TAMBAHAN (optional untuk CI)
# ===============================================================
artifacts_dir = os.path.join(base_path, "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

vectorizer_path = os.path.join(artifacts_dir, "vectorizer_ci.pkl")
joblib.dump(vectorizer, vectorizer_path)

report_path = os.path.join(artifacts_dir, "metrics_report_ci.txt")
with open(report_path, "w") as f:
    f.write("=== CI METRICS REPORT ===\n")
    f.write(f"Accuracy  : {acc:.4f}\n")
    f.write(f"Precision : {prec:.4f}\n")
    f.write(f"Recall    : {rec:.4f}\n")
    f.write(f"F1-score  : {f1:.4f}\n\n")
    f.write(classification_report(y_test, y_pred))

print("🎉 Training selesai tanpa error.")
print("==============================================")
