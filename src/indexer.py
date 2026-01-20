import os
import sys

# Ajout du chemin racine au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils.data_loader import load_cultural_events
from src.utils.vector_store import create_and_save_index

# Configuration des chemis
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "evenements_culturels.csv")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

def main():
    print("=== Démarrage de l'indexation ===")
    
    # Chargement des données brutes
    print(f"Chargement des données depuis {DATA_FILE}...")
    try:
        raw_docs = load_cultural_events(DATA_FILE)
    except FileNotFoundError:
        print(f"Erreur: Le fichier {DATA_FILE} n'existe pas. Lancez d'abord collect_data.py pour peupler la source de données.")
        return

    # Préparation des Documents LangChain
    # Enrichissement du contenu sémantique par concaténation des champs clés
    lc_docs = []
    for doc in raw_docs:
        # Format "Clé: Valeur" optimisé pour la compréhension du modèle d'embedding.
        content = f"Titre: {doc.get('title', '')}\nDescription: {doc.get('full_description', '')}\nDate: {doc.get('date_string', '')}\nLieu: {doc.get('location', '')}"
        metadata = {
            "uid": doc.get("uid"),
            "title": doc.get("title"),
            "date": doc.get("date_string"),
            "location": doc.get("location"),
            "source": "OpenAgenda"
        }
        lc_docs.append(Document(page_content=content, metadata=metadata))

    print(f"{len(lc_docs)} documents chargés.")

    # Chunking : Découpage sémantique (TextSplitter)
    print("Découpage des textes (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200 
    )
    chunks = text_splitter.split_documents(lc_docs)
    print(f"Généré {len(chunks)} chunks.")

    # Indexation et sauvegarde FAISS
    create_and_save_index(chunks, INDEX_DIR)
    print("=== Indexation terminée avec succès ===")

if __name__ == "__main__":
    main()
