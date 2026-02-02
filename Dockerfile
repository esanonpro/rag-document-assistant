# Dockerfile

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copier le code et les données
COPY app ./app
COPY data ./data
# Ne pas copier faiss_index, il sera créé au premier démarrage
# COPY faiss_index ./faiss_index

# Port exposé par l'API
EXPOSE 8000

# Commande de lancement
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
