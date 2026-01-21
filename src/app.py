import streamlit as st
import os
import sys

# Ajout du chemin racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import MISTRAL_API_KEY
from src.utils.vector_store import load_index
from langchain_mistralai import ChatMistralAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configuration de la page
st.set_page_config(page_title="Puls-Events Assistant", page_icon="🎭")

@st.cache_resource
def get_rag_chain():
    """
    Initialise la chaîne RAG (Retrieval-Augmented Generation).
    
    Args:
        None
    Returns:
        langchain.chains.RetrievalQA: La chaîne RAG initialisée.
    """
    # Chemin absolu pour éviter les erreurs de path selon le dossier d'exécution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(base_dir, "faiss_index")
    
    if not os.path.exists(index_path):
        st.error(f"Index non trouvé à : {index_path}. Veuillez exécuter 'src/indexer.py' pour générer la base vectorielle.")
        return None

    # Chargement de l'index FAISS (retriever configuration)
    vector_store = load_index(index_path)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Configuration du LLM (Mistral Large)
    # Temperature basse pour garantir la fiabilité factuelle
    llm = ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model="mistral-large-latest",
        temperature=0.2
    )

    # Prompt Système : Définition du rôle et des contraintes RAG
    system_prompt = (
        "Tu es **Puls-Assistant**, l'intelligence artificielle officielle de la plateforme culturelle Puls-Events.\n"
        "\n"
        "**Ton Objectif :** Recommander des sorties culturelles aux utilisateurs en te basant **uniquement** sur les informations d'événements fournies dans le CONTEXTE.\n"
        "\n"
        "Règles impératives (Garde-fous):\n"
        "1. **Fiabilité :** Tu ne dois jamais inventer d'événements. Si l'information n'est pas dans le contexte fourni, dis poliment que tu ne trouves pas d'événement correspondant dans ta base actuelle.\n"
        "2. **Ton :** Adopte un ton enthousiaste, professionnel et incitatif, comme un guide culturel passionné.\n"
        "3. **Format :** Pour chaque recommandation, précise toujours : Le titre, la date, le lieu, et pourquoi cela correspond à la demande de l'utilisateur.\n"
        "4. **Gestion des manques :** Si la demande est vague (ex: 'Je veux sortir'), pose une question de clarification (ex: 'Quel type d'ambiance cherchez-vous ou dans quelle ville ?').\n"
        "\n"
        "**CONTEXTE (Informations issues de la base vectorielle) :**\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Création de la chaîne RAG
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

st.title("🎭 Puls-Events Assistant")
st.markdown("Posez vos questions sur les événements culturels en Île-de-France !")

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
if query := st.chat_input("Ex: Quels concerts de jazz ce week-end ?"):
    # Ajout message utilisateur
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Génération réponse
    chain = get_rag_chain()
    
    if chain:
        with st.chat_message("assistant"):
            with st.spinner("Recherche d'événements..."):
                try:
                    response = chain.invoke({"input": query})
                    answer = response["answer"]
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Affichage limité des sources (max 3, format nettoyé)
                    with st.expander("Sources utilisées"):
                        for i, doc in enumerate(response["context"][:3]):
                            title = doc.metadata.get('title') or 'Sans titre'
                            date = doc.metadata.get('date') or ''
                            st.markdown(f"**Source {i+1}**: {title} ({date})")
                            snippet = doc.page_content[:200].replace('\n', ' ') + "..."
                            st.text(snippet)
                            
                except Exception as e:
                    st.error(f"Erreur lors de la génération : {e}")
