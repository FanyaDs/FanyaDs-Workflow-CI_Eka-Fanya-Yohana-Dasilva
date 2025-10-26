import os, sys, pandas as pd, joblib, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===============================================================
# [1] SETUP PATH DAN VALIDASI DATASET
# ===============================================================
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "namadataset_preprocessing", "wisata_bali_preprocessed.csv")

    print("==============================================")
    print("📂 Current Working Directory:", os.getcwd())
    print("📁 Base Path:", base_path)
    print(f"🔍 Mencoba memuat dataset dari: {data_path}")
    print("==============================================")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di path: {data_path}")

    df = pd.read_csv(data_path)
    if "clean_text" not in df.columns or "label" not in df.columns:
        raise KeyError("Kolom 'clean_text' atau 'label' tidak ditemukan di dataset!")

    df = df.dropna(subset=["clean_text", "label"])
    df["clean_text"] = df["clean_text"].astype(str).str.strip()
    print(f"✅ Dataset berhasil dimuat. Jumlah data: {len(df)} baris")

except Exception as e:
    print(f"❌ Gagal memuat dataset: {e}")
    sys.exit(1)

# ===============================================================
# [2] SPLIT DATA
# ===============================================================
try:
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42
    )
    print("✅ Dataset berhasil dibagi menjadi data train dan test.")
except Exception as e:
    print(f"❌ Gagal saat split data: {e}")
    sys.exit(1)

# ===============================================================
# [3] VEKTORISASI TEKS
# ===============================================================
try:
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("✅ Proses vektorisasi teks selesai.")
except Exception as e:
    print(f"❌ Gagal saat vektorisasi: {e}")
    sys.exit(1)

# ===============================================================
# [4] SETUP MLFLOW EXPERIMENT
# ===============================================================
try:
    mlflow.set_tracking_uri("file://" + os.path.join(base_path, "mlruns"))
    mlflow.set_experiment("Workflow_CI_Fanya")
    print("✅ MLflow Tracking URI dan experiment berhasil diinisialisasi.")
except Exception as e:
    print(f"❌ Gagal setup MLflow: {e}")
    sys.exit(1)

# ===============================================================
# [5] TRAINING DAN LOGGING MODEL
# ===============================================================
try:
    with mlflow.start_run() as run:
        print(f"🚀 Training model RandomForest dimulai... Run ID: {run.info.run_id}")

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

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
        joblib.dump(vectorizer, os.path.join(artifacts_dir, "vectorizer.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "vectorizer.pkl"))

        report_path = os.path.join(artifacts_dir, "metrics_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Accuracy  : {acc:.4f}\n")
            f.write(f"Precision : {prec:.4f}\n")
            f.write(f"Recall    : {rec:.4f}\n")
            f.write(f"F1-score  : {f1:.4f}\n")
        mlflow.log_artifact(report_path)

        print(f"✅ Model dan metrik berhasil dilog di Run ID: {run.info.run_id}")

except Exception as e:
    print(f"❌ Terjadi error saat training/logging: {e}")
    sys.exit(1)

print("🎉 Training selesai tanpa error.")
print("==============================================")
