# 🧪 RAG Science Assistant — Système de Recherche Documentaire Intelligent

> [!IMPORTANT]
> **🔒 Confidentialité & Données de Démonstration :**  
> Pour des raisons de confidentialité, les documents académiques originaux de l'**ENSAI** ne sont pas exposés dans ce dépôt public.  
> Pour la démonstration, le système a été alimenté avec un corpus d'**articles scientifiques publics** portant sur la **détection d'anomalies**, permettant ainsi de tester toutes les fonctionnalités de recherche et de citation sans compromettre de données sensibles.

## 🎯 Aperçu du Projet
Ce projet est un assistant de recherche capable d'extraire des informations pertinentes depuis un corpus de documents scientifiques volumineux. Il utilise la technique du **RAG (Retrieval-Augmented Generation)** pour fournir des réponses précises, sourcées et sans hallucinations.

### ❓ Pourquoi ce projet ?
L'IA générative classique (LLM) peut "halluciner" si elle n'a pas accès à un contexte spécifique. Ce système garantit :
- **Précision** : Les réponses sont basées uniquement sur des documents réels.
- **Transparence** : Chaque affirmation est accompagnée d'une citation directe de la source (Page, Extrait).
- **Flexibilité** : Fonctionne avec n'importe quel ensemble de PDFs.

## 🛠️ Architecture Technique
Le système repose sur une stack moderne et performante :
- **Extraction & Parsing** : Traitement des PDFs via LangChain (`PyPDFLoader`).
- **Indexation Vectorielle** : Utilisation de **FAISS** (Facebook AI Similarity Search) pour la recherche sémantique ultra-rapide.
- **Embeddings** : Modèle `all-MiniLM-L6-v2` de HuggingFace (pour une utilisation locale efficace).
- **Cerveau (LLM)** : **Mistral-7B** via **Ollama** pour une exécution 100% locale et privée.
- **Interface & API** : **FastAPI** pour une API robuste et documentée (Swagger UI).

## 🚀 Installation & Lancement Local

### 1. Prérequis
- **Python 3.11+**
- **Ollama** installé sur votre machine ([ollama.com](https://ollama.com))

### 2. Configuration d'Ollama
Téléchargez le modèle Mistral :
```bash
ollama pull mistral
```

### 3. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 4. Lancement de l'API
Ouvrez un terminal à la racine du projet et lancez :
```bash
uvicorn app.api:app --reload
```
L'interface Swagger sera alors accessible sur : `http://localhost:8000/docs`.

## 📂 Structure du Projet
- `app/` : Code source de l'API et du pipeline RAG.
- `data/` : Dossier contenant les documents PDFs à indexer.
- `faiss_index/` : Stockage local de l'index vectoriel.
- `requirements.txt` : Liste des dépendances Python.

## 🤝 Contribution
Ce projet a été développé dans le cadre d'un portfolio pour démontrer des compétences en **NLP**, **Vector Databases** et **Architecture LLM**.
