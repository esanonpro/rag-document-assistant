# - RAG Science Assistant — Système de Recherche Documentaire Intelligent

> [!IMPORTANT]
> **- Confidentialité & Données de Démonstration :**  
> Pour des raisons de confidentialité, les documents académiques originaux de l'**ENSAI** ne sont pas exposés dans ce dépôt public.  
> Pour la démonstration, le système a été alimenté avec un corpus d'**articles scientifiques publics** portant sur la **détection d'anomalies**, permettant ainsi de tester toutes les fonctionnalités de recherche et de citation sans compromettre de données sensibles.

## - Aperçu du Projet
Ce projet est un assistant de recherche capable d'extraire des informations pertinentes depuis un corpus de documents scientifiques volumineux. Il utilise la technique du **RAG (Retrieval-Augmented Generation)** pour fournir des réponses ancrées dans le corpus et accompagnées de leurs sources.

### - Pourquoi ce projet ?
L'IA générative classique (LLM) peut "halluciner" si elle n'a pas accès à un contexte spécifique. Le système apporte les éléments suivants :
- **Ancrage documentaire** : le pipeline fournit au modèle des passages récupérés dans le corpus avant la génération.
- **Transparence** : Les passages récupérés sont exposés avec leur document, leur page et un extrait ; cela ne prouve pas à lui seul que chaque affirmation est supportée.
- **Flexibilité** : Fonctionne avec n'importe quel ensemble de PDFs.

## - Architecture Technique
Le système repose sur une stack moderne et performante :
- **Extraction & Parsing** : Traitement des PDFs via LangChain (`PyPDFLoader`).
- **Indexation Vectorielle** : Utilisation de **FAISS** (Facebook AI Similarity Search) pour la recherche sémantique ultra-rapide.
- **Embeddings** : Modèle `all-MiniLM-L6-v2` de HuggingFace (pour une utilisation locale efficace).
- **Cerveau (LLM)** : **Mistral-7B** via **Ollama** pour une exécution 100% locale et privée.
- **Interface & API** : **FastAPI** pour une API robuste et documentée (Swagger UI).


## - Évaluation du système RAG

L'évaluation repose sur **RAGAS** avec trois dimensions volontairement complémentaires. L'objectif est de couvrir les trois points critiques du pipeline sans multiplier les métriques redondantes.

| Dimension | Métrique | Question évaluée |
| --- | --- | --- |
| Retrieval | **Context Recall** | Le retriever fournit-il au LLM les informations nécessaires pour répondre ? |
| Ancrage | **Faithfulness** | Les affirmations générées sont-elles supportées par le contexte récupéré ? |
| Réponse | **Answer Relevancy** | La réponse traite-t-elle réellement la question posée ? |

### Pourquoi ces trois métriques ?

- **Context Recall** est prioritaire sur des métriques de ranking comme MRR : dans ce RAG, les passages du top-k sont transmis automatiquement au LLM ; l'enjeu principal est donc que l'information nécessaire soit présente dans le contexte.
- **Faithfulness** mesure un risque propre aux systèmes génératifs : produire une affirmation qui n'est pas supportée par les documents récupérés.
- **Answer Relevancy** complète la faithfulness : une réponse peut être fidèle au contexte tout en étant peu pertinente pour la question.

RAGAS fournit le cadre d'évaluation ; certaines métriques de génération reposent sur une évaluation sémantique de type **LLM-as-a-Judge**. Le juge n'est pas considéré comme une vérité terrain : le modèle évaluateur, le prompt et la configuration doivent être conservés avec les résultats.

### Protocole reproductible

1. Utiliser les 12 questions de référence vérifiées contre cinq des 11 PDF fournis (documents et pages dans le benchmark).
2. Associer aux questions les réponses ou contextes de référence nécessaires à l'évaluation.
3. Exécuter le pipeline avec une configuration figée (chunking, `top_k`, embeddings et LLM).
4. Calculer **Context Recall**, **Faithfulness** et **Answer Relevancy** avec RAGAS.
5. Conserver la configuration avec les scores pour rendre les comparaisons reproductibles.

Le script reproductible est disponible dans `evaluation/evaluate_rag.py`. Il génère `evaluation/results.json` à partir d'un jeu de questions/réponses de référence revu manuellement.

```bash
python -m evaluation.evaluate_rag --corpus data \
  --rag-model mistral-eval-v03 --judge-model mistral-eval-v03 \
  --embedding-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --output evaluation/rerun
```

> **Résultats : non publiés pour le moment.** Le benchmark contient désormais 12 entrées revues et sourcées. Les PDF ne sont pas redistribués. Aucun score synthétique ne remplace une mesure réelle.

Le [protocole détaillé](evaluation/README.md) précise l'environnement figé, l'import GGUF dans Ollama, les rôles du générateur et du juge, les traces et les limites de l'évaluation.

## - Installation & Lancement Local

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

## - Structure du Projet
- `app/` : Code source de l'API et du pipeline RAG.
- `data/` : Dossier contenant les documents PDFs à indexer.
- `faiss_index/` : Stockage local de l'index vectoriel.
- `requirements.txt` : Liste des dépendances Python.

## - Contribution
Ce projet a été développé dans le cadre d'un portfolio pour démontrer des compétences en **NLP**, **Vector Databases** et **Architecture LLM**.
