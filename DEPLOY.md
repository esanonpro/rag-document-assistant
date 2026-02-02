# Guide de Déploiement - RAG Science App

Ce projet est configuré pour être déployé via **Docker**. Il contient un `Dockerfile` prêt à l'emploi.

## 1. Prérequis

- **Docker** installé sur votre machine.
- Une clé API OpenAI valide (`OPENAI_API_KEY`).

## 2. Déploiement Local (Docker)

C'est la méthode la plus simple pour tester l'application.

### Étape 1 : Construire l'image
Ouvrez un terminal à la racine du projet et exécutez :

```bash
docker build -t rag-science-api .
```

### Étape 2 : Lancer le conteneur
Vous devez passer votre clé API en variable d'environnement.

**Option A : Via la ligne de commande**
```bash
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-proj-xxxx rag-science-api
```
1.  Avoir un compte Google Cloud et un projet actif.
2.  Installer le [Google Cloud SDK (gcloud CLI)](https://cloud.google.com/sdk/docs/install).

### Étape 1 : Initialisation
Connectez-vous et configurez votre projet :
```bash
gcloud auth login
gcloud config set project VOTRE_ID_PROJET
```

### Étape 2 : Activer les services nécessaires
```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

### Étape 3 : Déployer depuis la source
Google Cloud Run peut construire votre conteneur automatiquement à partir du code source.

```bash
gcloud run deploy rag-science-api \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=votre_clé_api_ici
```

*Note : Remplacez `votre_clé_api_ici` par votre vraie clé. Pour la production, utilisez le Secret Manager.*

### Étape 4 : Accéder à l'API
La commande affichera une URL (ex: `https://rag-science-api-xxxxx-ew.a.run.app`).
Vous pouvez tester :
- Swagger : `https://.../docs`

## 4. Autres Options (Fly.io, etc.)
Si vous préférez Fly.io :
1. `fly launch`
2. `fly secrets set OPENAI_API_KEY=...`
3. `fly deploy`

## 5. Automatisation (CI/CD)

Le fichier `.github/workflows/ci-cd.yml` est actuellement vide.
Si vous souhaitez mettre en place un déploiement automatique (par exemple, déployer sur Fly.io à chaque `git push`), nous pouvons configurer ce fichier.

Dites-moi si vous souhaitez que je configure le pipeline CI/CD !
