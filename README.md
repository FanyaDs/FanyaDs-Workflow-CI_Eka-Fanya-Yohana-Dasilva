🚀 Workflow CI – MLflow Project

Repository ini dibuat untuk memenuhi Kriteria 3: Workflow CI (Continuous Integration) pada proyek akhir kelas Dicoding – Membangun Sistem Machine Learning (MSML).
Workflow ini menggunakan MLflow Project dan GitHub Actions untuk melakukan pelatihan ulang model secara otomatis, menyimpan artefak hasil pelatihan, serta membangun dan mengunggah Docker image ke Docker Hub.

🎯 Tujuan

Melatih ulang model secara otomatis setiap kali terjadi perubahan pada branch main.

Melacak eksperimen dan metrik menggunakan MLflow Tracking.

Menyimpan artefak hasil pelatihan ke GitHub Actions Artifact.

Membangun dan mengunggah Docker image model ke Docker Hub.

⚙️ Alur Workflow

1. Setup Environment
Menginstal dependensi Python seperti mlflow, scikit-learn, pandas, dagshub, docker, dan joblib.

2. Training Model
Menjalankan file modelling.py untuk melatih model dan mencatat metrik ke MLflow Tracking.

3. Upload Artefak
Mengunggah folder mlruns ke GitHub Actions Artifact agar hasil pelatihan tersimpan otomatis.

4. Build Docker Image
Menggunakan perintah mlflow build-docker untuk membangun Docker image dari model yang telah dilatih.

5. Push ke Docker Hub
Mengunggah Docker image ke repository Docker Hub menggunakan kredensial rahasia (DOCKER_USERNAME dan DOCKER_PASSWORD).

🐳 Docker Hub

Repository: https://hub.docker.com/repository/docker/fanyads/fanya-mlflow-model

Nama Image: fanyads/fanya-mlflow-model:latest

🏁 Pencapaian Kriteria 3
Level	Deskripsi	Status
Basic (2 pts)	Workflow otomatis melatih model	✅ Terpenuhi
Skilled (3 pts)	Menyimpan artefak model ke repo/Drive	✅ Terpenuhi
Advanced (4 pts)	Build dan push Docker image ke Docker Hub	✅ Terpenuhi
👩‍💻 Author

Nama: Eka Fanya Yohana Dasilva
Dicoding ID: M401D5X0523
Institusi: ITN Malang – Teknik Informatika
