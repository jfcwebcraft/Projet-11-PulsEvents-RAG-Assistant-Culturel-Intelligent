import streamlit as st
import os
import sys
import pandas as pd

# Ajout du chemin racine
# Ajout du chemin racine
# On est dans src/pages/ donc il faut remonter de 3 crans pour arriver à la racine du projet
# __file__ = src/pages/script.py
# dirname 1 = src/pages
# dirname 2 = src
# dirname 3 = racine
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from src.utils.vector_store import load_index

st.set_page_config(page_title="Explorateur de Base", page_icon="🗂️", layout="wide")

st.title("🗂️ Explorateur de Base Vectorielle")
st.markdown("Cet outil permet d'inspecter le contenu brut indexé dans FAISS pour le débogage et la validation.")

# Chargement de l'index
index_path = os.path.join(root_dir, "faiss_index")

if not os.path.exists(index_path):
    st.error("Index non trouvé !")
else:
    try:
        vector_store = load_index(index_path)
        
        # Accès direct aux documents stockés (Hack pour FAISS)
        # On récupère le dictionnaire interne {uuid: Document}
        docs_dict = vector_store.docstore._dict
        
        st.metric(label="Nombre total de chunks", value=len(docs_dict))
        
        # Conversion en DataFrame pour affichage
        data = []
        for doc_id, doc in docs_dict.items():
            row = {
                "ID": doc_id,
                "Titre": doc.metadata.get("title", "N/A"),
                "Date": doc.metadata.get("date", "N/A"),
                "Lieu": doc.metadata.get("location", "N/A"),
                "Source": doc.metadata.get("source", "N/A"),
                "Contenu Texte (Chunk)": doc.page_content
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Filtres
        search = st.text_input("🔍 Rechercher dans la base (Titre ou Contenu)", "")
        
        if search:
            df = df[
                df["Titre"].str.contains(search, case=False, na=False) | 
                df["Contenu Texte (Chunk)"].str.contains(search, case=False, na=False)
            ]
            
        st.dataframe(
            df, 
            use_container_width=True,
            column_config={
                "Contenu Texte (Chunk)": st.column_config.TextColumn("Contenu", width="large"),
            }
        )
        
        with st.expander("ℹ️ Détails techniques"):
            st.info("Ces données proviennent du fichier `index.pkl` local associé à `index.faiss`.")

    except Exception as e:
        st.error(f"Erreur de lecture de l'index : {e}")
