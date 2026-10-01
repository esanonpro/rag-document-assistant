# - RAG Science Assistant — Système de Recherche Documentaire Intelligent

> [!IMPORTANT]
> **- Confidentialité & Données de Démonstration :**  
> Pour des raisons de confidentialité, les documents académiques originaux de l'**ENSAI** ne sont pas exposés dans ce dépôt public.  
> Pour la démonstration, le système a été alimenté avec un corpus d'**articles scientifiques publics** portant sur la **détection d'anomalies**, permettant ainsi de tester toutes les fonctionnalités de recherche et de citation sans compromettre de données sensibles.

## - Aperçu du Projet
Ce projet est un assistant de recherche capable d'extraire des informations pertinentes depuis un corpus de documents scientifiques volumineux. Il utilise la technique du **RAG (Retrieval-Augmented Generation)** pour fournir des réponses ancrées dans le corpus et accompagnées de leurs sources.

### - Pourquoi ce projet ?
L'IA générative classique (LLM) peut "halluciner" si elle n'a pas accès à un contexte spécifique. Ce système garantit :
- **Ancrage documentaire** : le pipeline fournit au modèle des passages récupérés dans le corpus avant la génération.
- **Transparence** : Chaque affirmation est accompagnée d'une citation directe de la source (Page, Extrait).
- **Flexibilité** : Fonctionne avec n'importe quel ensemble de PDFs.

## - Architecture Technique
Le système repose sur une stack moderne et performante :
- **Extraction & Parsing** : Traitement des PDFs via LangChain (`PyPDFLoader`).
- **Indexation Vectorielle** : Utilisation de **FAISS** (Facebook AI Similarity Search) pour la recherche sémantique ultra-rapide.
- **Embeddings** : Modèle `all-MiniLM-L6-v2` de HuggingFace (pour une utilisation locale efficace).
- **Cerveau (LLM)** : **Mistral-7B** via **Ollama** pour une exécution 100% locale et privée.
- **Interface & API** : **FastAPI** pour une API robuste et documentée (Swagger UI).


## - Évaluation du système RAG

L'évaluation est pensée en deux niveaux complémentaires afin de distinguer la qualité du **retrieval** de celle de la **réponse générée**. Les scores ne sont pas publiés tant qu'un jeu de questions de référence annoté n'a pas été constitué.

### 1. Évaluation du retrieval

À partir d'un jeu de questions pour lesquelles les documents ou passages pertinents sont connus :

- **Recall@k** : proportion de questions pour lesquelles au moins un passage pertinent apparaît dans les `k` premiers résultats.
- **Precision@k** : proportion de passages pertinents parmi les `k` passages retournés.
- **MRR (Mean Reciprocal Rank)** : mesure la position du premier passage pertinent ; elle pénalise un document pertinent retrouvé trop bas dans le classement.
- **Hit Rate@k** : proportion de requêtes ayant au moins un document pertinent dans le top-k.

Ces métriques permettent notamment de comparer les choix de chunking, la valeur de `k`, le modèle d'embeddings et la configuration de l'index FAISS.

### 2. Évaluation de la génération

Sur un ensemble de questions/réponses de référence :

- **Faithfulness / groundedness** : vérifie si les affirmations de la réponse sont effectivement supportées par le contexte récupéré.
- **Answer relevance** : mesure dans quelle mesure la réponse traite directement la question posée.
- **Context relevance** : évalue si les passages transmis au LLM sont utiles pour répondre à la question.
- **Exact Match / F1** : utilisables lorsque les questions disposent d'une réponse de référence suffisamment courte et non ambiguë.
- **Évaluation humaine** : contrôle complémentaire de la justesse, de la qualité des citations et de l'utilité de la réponse.

### 3. Protocole

Le protocole cible un jeu de questions annotées couvrant plusieurs articles et plusieurs niveaux de difficulté. Les résultats seront reportés avec la configuration évaluée (chunking, `top_k`, embeddings et LLM) afin de rendre les comparaisons reproductibles.

> **Résultats : à compléter après constitution et annotation du jeu d'évaluation.**

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
