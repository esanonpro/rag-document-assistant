import streamlit as st
import time
from rag_pipeline import create_rag_chain, build_or_load_vectorstore, VECTORSTORE_PATH

# Configuration de la page
st.set_page_config(
    page_title="RAG Science Assistant",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Assistant de Recherche Scientifique")
st.markdown("""
Posez vos questions sur les documents scientifiques indexés. 
L'assistant utilise **Mistral-7B** via l'API Hugging Face.
""")

# Initialisation de l'état de session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    with st.spinner("Chargement du moteur de recherche..."):
        # On charge le vectorstore et la chaine une seule fois
        vectorstore = build_or_load_vectorstore(VECTORSTORE_PATH)
        retriever, chain = create_rag_chain(vectorstore)
        st.session_state.retriever = retriever
        st.session_state.rag_chain = chain
        st.success("Moteur prêt !")

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
if prompt := st.chat_input("Votre question scientifique..."):
    # Afficher la question utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Générer la réponse
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # 1. Récupération de la réponse
            with st.spinner("Analyse des documents en cours..."):
                response = st.session_state.rag_chain.invoke(prompt)
            
            # 2. Récupération des sources
            docs = st.session_state.retriever.invoke(prompt)
            
            # Simulation de stream (optionnel, pour l'effet visuel)
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.02)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            # 3. Affichage des sources
            with st.expander("📚 Sources utilisées"):
                for i, doc in enumerate(docs):
                    source_name = doc.metadata.get("source", "Inconnu")
                    page_num = doc.metadata.get("page", "?")
                    st.markdown(f"**Source {i+1}** : `{source_name}` (Page {page_num})")
                    st.caption(f"...{doc.page_content[:300]}...")

            # Sauvegarde dans l'historique
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
