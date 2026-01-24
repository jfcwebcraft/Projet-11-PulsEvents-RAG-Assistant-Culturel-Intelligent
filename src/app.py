import streamlit as st
import os
import sys

# Ajout du chemin racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import get_rag_chain as engine_get_rag_chain

# Configuration de la page
st.set_page_config(page_title="Puls-Events Assistant", page_icon="🎭")

@st.cache_resource
def get_rag_chain():
    """
    Wrapper pour initialiser la chaîne RAG via le moteur dédié.
    Utilise le cache Streamlit pour ne pas recharger le modèle à chaque interaction.
    """
    try:
        return engine_get_rag_chain()
    except FileNotFoundError as e:
        st.error(str(e))
        return None
    except Exception as e:
        st.error(f"Erreur d'initialisation du moteur RAG : {e}")
        return None

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
