import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
DATASET_ID = os.getenv("HF_DATASET_ID")  # ex: ElieSan99/rag-science-data

if not HF_TOKEN:
    raise ValueError("❌ HUGGINGFACEHUB_API_TOKEN manquant dans .env")

if not DATASET_ID:
    dataset_name = input("Nom du nouveau Dataset (ex: ElieSan99/rag-science-data) : ")
    DATASET_ID = dataset_name

api = HfApi(token=HF_TOKEN)

def upload_data():
    print(f"🚀 Préparation de l'upload vers {DATASET_ID}...")

    # 1. Créer le repo s'il n'existe pas
    try:
        create_repo(DATASET_ID, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
        print(f"✅ Dataset {DATASET_ID} prêt.")
    except Exception as e:
        print(f"⚠️ Erreur création repo (il existe peut-être déjà) : {e}")

    # 2. Upload du dossier data
    print("📤 Upload du dossier 'data'...")
    api.upload_folder(
        folder_path="data",
        path_in_repo="data",
        repo_id=DATASET_ID,
        repo_type="dataset"
    )

    # 3. Upload du dossier faiss_index
    print("📤 Upload du dossier 'faiss_index'...")
    api.upload_folder(
        folder_path="faiss_index",
        path_in_repo="faiss_index",
        repo_id=DATASET_ID,
        repo_type="dataset"
    )

    print("🎉 Upload terminé avec succès !")
    print(f"👉 Retrouvez vos données ici : https://huggingface.co/datasets/{DATASET_ID}")

if __name__ == "__main__":
    upload_data()
