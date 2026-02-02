# app/rag_pipeline.py

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.llms import Ollama

load_dotenv()

DATA_DIR = Path("data")

VECTORSTORE_PATH = Path("faiss_index")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")


# --------- 1. Chargement & chunking ---------

def load_pdfs_from_folder(folder: Path):
    docs = []
    for pdf_path in folder.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    return docs


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# --------- 2. Embeddings & FAISS ---------

def create_embedding_function():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_or_load_vectorstore(persist_path: Path):
    """
    Si un index FAISS valide existe -> on le charge.
    Sinon -> on charge les PDFs, on chunk, on embed, on construit FAISS.
    """
    embedding_function = create_embedding_function()

    index_file = persist_path / "index.faiss"
    store_file = persist_path / "index.pkl"

    # Cas 1 : un index FAISS existe déjà -> on le recharge
    if index_file.exists() and store_file.exists():
        print("🔁 Chargement de l'index FAISS existant...")
        vectorstore = FAISS.load_local(
            str(persist_path),
            embeddings=embedding_function,
            allow_dangerous_deserialization=True,
        )
        return vectorstore

    # Cas 2 : rien n'existe -> on le construit
    print("⚙️ Aucun index FAISS trouvé, construction en cours...")

    raw_docs = load_pdfs_from_folder(DATA_DIR)
    if not raw_docs:
        raise ValueError(f"Aucun PDF trouvé dans le dossier {DATA_DIR}. "
                         "Ajoute au moins un fichier .pdf pour initialiser l'index.")

    chunks = split_documents(raw_docs)
    print(f"✅ {len(chunks)} chunks générés à partir des PDF.")

    persist_path.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(chunks, embedding_function)
    vectorstore.save_local(str(persist_path))
    print(f"💾 Nouvel index FAISS sauvegardé dans {persist_path}")

    return vectorstore


# --------- 3. LLM ---------


def create_llm(model_name: str = "mistral", temperature: float = 0.2):
    """
    Crée un LLM local via Ollama.
    Exemple de modèles disponibles : mistral, llama3, phi3, codellama, gemma
    """
    return Ollama(model=model_name, temperature=temperature)



# --------- 4. RAG chain (sans RetrievalQA) ---------

def format_docs(docs):
    """Concatène le contenu des documents pour le passer au LLM."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(vectorstore=None):
    """
    Construit la chaîne RAG :
    - retriever -> docs
    - prompt (context + question)
    - LLM
    - parser (string)
    Retourne (retriever, rag_chain) pour qu'on puisse aussi récupérer les sources.
    """
    if vectorstore is None:
        vectorstore = build_or_load_vectorstore(VECTORSTORE_PATH)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    llm = create_llm("mistral")

    system_template = (
        "Tu es un assistant de recherche scientifique rigoureux. "
        "Tu réponds en t'appuyant UNIQUEMENT sur le contexte fourni.\n\n"
        "Si tu ne trouves pas la réponse dans le contexte, dis explicitement "
        "\"Je ne sais pas à partir des documents fournis\".\n\n"
        "Contexte :\n{context}\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", "Question : {question}\n\nRéponse détaillée :"),
        ]
    )

    # Chaîne RAG avec LCEL
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, rag_chain
