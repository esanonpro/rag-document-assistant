# app/api.py

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List

from .rag_pipeline import (
    build_or_load_vectorstore,
    create_rag_chain,
    VECTORSTORE_PATH,
)

app = FastAPI(
    title="Scientific RAG Assistant API",
    description="Assistant de recherche scientifique basé sur RAG + LLM",
    version="1.0.0",
)

# Initialisation globale (chargée au démarrage du service)
vectorstore = build_or_load_vectorstore(VECTORSTORE_PATH)
retriever, rag_chain = create_rag_chain(vectorstore)  # 🔴 ICI : on récupère les 2


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    source: str
    page: int
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    question = payload.question

    # 1) Appel de la chaîne RAG → réponse (string)
    answer = rag_chain.invoke(question)

    # 2) Récupération des sources via le retriever
    docs = retriever.invoke(question)

    sources = []
    for doc in docs[: payload.top_k]:
        meta = doc.metadata
        sources.append(
            Source(
                source=str(meta.get("source", "unknown")),
                page=int(meta.get("page", -1)),
                snippet=doc.page_content[:300],
            )
        )

    return QueryResponse(
        answer=answer,
        sources=sources,
    )
