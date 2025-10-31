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
# [1] SETUP PATH DAN VALIDASI DATASET
# ===============================================================
try:
    base_path = os.getcwd()
    data_path = os.path.join(base_path, "namadataset_preprocessing", "wisata_bali_preprocessed.csv")

    print(f"🔍 Mencoba memuat dataset dari: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di path: {data_path}")

    df = pd.read_csv(data_path)

    if "clean_text" not in df.columns or "label" not in df.columns:
        raise KeyError("Kolom 'clean_text' atau 'label' tidak ditemukan di dataset!")

    df = df.dropna(subset=["clean_text", "label"])
    df["clean_text"] = df["clean_text"].astype(str).str.strip()

    print(f"✅ Dataset berhasil dimuat ({len(df)} baris)")
    print(f"📊 Contoh data:\n{df.head(3)}")

except Exception as e:
    print(f"❌ Gagal memuat dataset: {e}")
    print("⚠️ [DEBUG] Program tidak dihentikan agar log tetap muncul.")
    df = pd.DataFrame({"clean_text": ["fallback"], "label": [0]})  # dummy data agar tidak error lanjut

# ===============================================================
# [2] SPLIT DATA
# ===============================================================
try:
    X = df["clean_text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ Dataset berhasil dibagi menjadi train/test.")
    print(f"📈 Data Train: {len(X_train)}, Data Test: {len(X_test)}")
except Exception as e:
    print(f"❌ Gagal split data: {e}")
    sys.exit(1)

# ===============================================================
# [3] VEKTORISASI TEKS
# ===============================================================
try:
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("✅ Vektorisasi teks selesai.")
    print(f"🧠 Jumlah fitur: {len(vectorizer.get_feature_names_out())}")
except Exception as e:
    print(f"❌ Gagal vektorisasi: {e}")
    sys.exit(1)

# ===============================================================
# [4] SETUP MLFLOW EXPERIMENT
# ===============================================================
try:
    mlruns_path = os.path.join(base_path, "mlruns")
    os.makedirs(mlruns_path, exist_ok=True)

    mlflow.set_tracking_uri("file://" + mlruns_path)
    mlflow.set_experiment("Workflow_CI_Fanya")
    print(f"✅ MLflow Tracking URI diinisialisasi di: {mlruns_path}")
except Exception as e:
    print(f"❌ Gagal setup MLflow: {e}")
    sys.exit(1)

# ===============================================================
# [5] TRAINING DAN LOGGING MODEL
# ===============================================================
try:
    print("🧭 Membuat run baru MLflow (mode CI)...")
    with mlflow.start_run(run_name="CI_Run_Fanya") as run:
        print(f"🚀 Training RandomForest dimulai... Run ID: {run.info.run_id}")

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)

        # === METRICS ===
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

        # === LOG KE MLFLOW ===
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("vectorizer", "CountVectorizer")

        # === SIMPAN MODEL ===
        mlflow.sklearn.log_model(clf, artifact_path="model")

        # === LOG ARTEFAK TAMBAHAN ===
        artifacts_dir = os.path.join(base_path, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        vectorizer_path = os.path.join(artifacts_dir, "vectorizer.pkl")
        joblib.dump(vectorizer, vectorizer_path)
        mlflow.log_artifact(vectorizer_path)

        report_path = os.path.join(artifacts_dir, "metrics_report.txt")
        with open(report_path, "w") as f:
            f.write("=== MODEL METRICS REPORT ===\n")
            f.write(f"Accuracy  : {acc:.4f}\n")
            f.write(f"Precision : {prec:.4f}\n")
            f.write(f"Recall    : {rec:.4f}\n")
            f.write(f"F1-score  : {f1:.4f}\n\n")
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact(report_path)

        print(f"✅ Model dan metrik berhasil dilog di Run ID: {run.info.run_id}")
except Exception as e:
    print(f"❌ Terjadi error saat training/logging: {e}")
    sys.exit(1)

# ===============================================================
# [6] SELESAI
# ===============================================================
print("🎉 Training selesai tanpa error.")
print("==============================================")
