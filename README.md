Workflow CI – MLflow Project

Repository ini dibuat untuk memenuhi Kriteria 3: Workflow CI (Continuous Integration) pada proyek akhir kelas Dicoding - Membangun Sistem Machine Learning (MSML).
Workflow ini menggunakan MLflow Project dan GitHub Actions untuk melakukan pelatihan ulang model secara otomatis, menyimpan artefak hasil pelatihan, dan membangun Docker image ke Docker Hub.

Tujuan

Melatih ulang model otomatis setiap kali terjadi perubahan di branch main.

Melacak eksperimen dan metrik dengan MLflow Tracking.

Menyimpan artefak hasil pelatihan ke GitHub Actions Artifact.

Membangun dan mengunggah Docker image model ke Docker Hub.


Alur Workflow

Setup Environment – Menginstal dependensi Python (mlflow, scikit-learn, pandas, dagshub, docker, joblib).

Training Model – Menjalankan modelling.py untuk melatih model dan mencatat metrik ke MLflow.

Upload Artefak – Mengunggah folder mlruns ke GitHub Actions Artifact.

Build Docker Image – Menggunakan mlflow build-docker untuk membuat image model.

Push ke Docker Hub – Mengunggah image model ke repository Docker Hub.

Docker Hub

Repository:
https://hub.docker.com/repository/docker/fanyads/fanya-mlflow-model

Nama image:
fanyads/fanya-mlflow-model:latest

Pencapaian Kriteria 3
Level	Deskripsi	Status
Basic (2 pts)	Workflow otomatis melatih model	Terpenuhi
Skilled (3 pts)	Menyimpan artefak model ke repo/Drive	Terpenuhi
Advanced (4 pts)	Build dan push Docker image ke Docker Hub	Terpenuhi

Author:
Eka Fanya Yohana Dasilva
Dicoding ID: M401D5X0523
ITN Malang – Teknik Informatika
