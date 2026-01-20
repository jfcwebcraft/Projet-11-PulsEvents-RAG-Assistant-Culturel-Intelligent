import os
import faiss
import numpy as np
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.utils.config import MISTRAL_API_KEY

def get_embeddings_model():
    """Retourne le modèle d'embedding Mistral configuré."""
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY manquante.")
    return MistralAIEmbeddings(api_key=MISTRAL_API_KEY, model="mistral-embed")

def create_and_save_index(chunks: list[Document], index_path: str):
    """
    Crée un index FAISS à partir des chunks et le sauvegarde.
    """
    embeddings = get_embeddings_model()
    
    print(f"Création de l'index FAISS pour {len(chunks)} chunks...")
    # Vectorisation et indexation via Mistral/FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print(f"Sauvegarde de l'index dans {index_path}...")
    vector_store.save_local(index_path)
    return vector_store

def load_index(index_path: str):
    """
    Charge un index FAISS existant.
    """
    embeddings = get_embeddings_model()
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index non trouvé à : {index_path}")
        
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store
