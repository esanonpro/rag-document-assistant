# 🧬 RAG Science App — Assistant de Recherche Documentaire Intelligent

> [!TIP]
> **🚀 Déploiement :** Ce projet est conçu pour être hébergé sur Hugging Face Spaces.  
> **Interface :** Streamlit

## 🎯 Contexte métier
Dans le domaine scientifique, la masse de documents à traiter est colossale. Ce projet vise à offrir aux chercheurs un assistant capable d'extraire des informations pertinentes depuis un corpus de PDFs scientifiques, en fournissant des réponses sourcées et précises.

## ❓ Problème à résoudre
L'IA générative classique (LLM) a tendance à "halluciner" si elle n'a pas accès à un contexte spécifique. Ce projet utilise la technique du **RAG (Retrieval-Augmented Generation)** pour :
- Garantir que les réponses sont basées uniquement sur des documents réels.
- Fournir des citations précises pour chaque affirmation.
- Permettre l'interrogation de documents non inclus dans l'entraînement initial de l'IA.

## 🏗️ Architecture & Stack Technique
Le système repose sur un pipeline moderne d'extraction et de recherche :
- **Extraction :** Traitement des PDFs via des scripts de parsing spécialisés.
- **Indexation :** Utilisation de **FAISS** (Vector Store) pour stocker les embeddings.
- **Modèles :** Mistral (via Hugging Face API) et LangChain pour l'orchestration du RAG.
- **Interface :** Streamlit pour une expérience utilisateur fluide et interactive.

## 🚀 Fonctionnalités
- **RAG Hybride** : Recherche documentaire sémantique combinée à la génération de texte.
- **Citations des Sources** : Chaque réponse inclut les extraits précis des documents utilisés.
- **Conteneurisation** : Docker Ready pour un déploiement robuste.

## 🛠️ Instructions Techniques
### Prérequis
- Python 3.11
- Un token Hugging Face (à configurer dans `.env`)

### Lancement Local
1. **Installation** : `pip install -r requirements.txt`
2. **Exécution** : `streamlit run app/main.py`
3. **Usage via Docker** : 
   - `docker build -t rag-science-app .`
   - `docker run -p 7860:7860 --env-file .env rag-science-app`
