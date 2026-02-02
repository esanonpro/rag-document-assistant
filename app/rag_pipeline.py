# app/rag_pipeline.py

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEndpoint # On n'utilise plus ça

from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from huggingface_hub import InferenceClient

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

DATA_DIR = Path("data")
VECTORSTORE_PATH = Path("faiss_index")

# Téléchargement automatique des données depuis le Dataset Hugging Face
HF_DATASET_ID = os.getenv("HF_DATASET_ID")
if HF_DATASET_ID:
    HF_DATASET_ID = HF_DATASET_ID.strip()
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        token = token.strip()
    
    try:
        snapshot_download(
            repo_id=HF_DATASET_ID,
            repo_type="dataset",
            local_dir=".",
            allow_patterns=["data/*", "faiss_index/*"],
            token=token
        )
        print("✅ Données téléchargées avec succès.")
    except Exception as e:
        print(f"⚠️ Erreur lors du téléchargement des données : {e}")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)

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

from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from huggingface_hub import InferenceClient
import traceback

class HFChatLLM(LLM):
    """
    Wrapper personnalisé pour utiliser l'API 'Chat Completion' de Hugging Face.
    Cela contourne les erreurs 'Task not supported' de l'API text-generation.
    """
    repo_id: str = "HuggingFaceH4/zephyr-7b-beta"
    token: str = ""

    @property
    def _llm_type(self) -> str:
        return "hf_chat_inference"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        client = InferenceClient(token=self.token)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = client.chat_completion(
                messages=messages,
                model=self.repo_id,
                max_tokens=512,
                temperature=0.1,
                top_p=0.95,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ ERROR DETAIL: {e}")
            traceback.print_exc()
            return f"Erreur lors de la génération : {e}"

def create_llm():
    """
    Crée notre LLM personnalisé qui utilise l'API Chat.
    """
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        token = token.strip()
        
    return HFChatLLM(token=token)


# --------- 4. RAG chain ---------

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
        search_kwargs={"k": 3},
    )

    llm = create_llm()

    # On fusionne tout dans un seul prompt utilisateur pour être sûr que le modèle le prenne en compte
    template = (
        "Instructions : Tu es un assistant qui répond UNIQUEMENT en utilisant le texte fourni ci-dessous.\n"
        "Tu ne dois JAMAIS utiliser tes connaissances générales.\n"
        "Si la réponse n'est pas dans le texte, réponds exactement : 'Je ne trouve pas l'information dans les documents fournis'.\n\n"
        "Texte de référence :\n{context}\n\n"
        "Question : {question}\n"
        "Réponse :"
    )

    prompt = ChatPromptTemplate.from_template(template)

    def clean_output(text: str) -> str:
        # Nettoyage des balises d'hallucination courantes
        text = text.split("[/USER]")[0]
        text = text.split("[/ASS]")[0]
        return text.strip()

    # Chaîne RAG avec LCEL
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
        | clean_output
    )

    return retriever, rag_chain
