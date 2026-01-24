import sys
import os

# Ajout du chemin racine pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.vector_store import load_index
from src.utils.config import MISTRAL_API_KEY

def inspect_faiss_content():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(base_dir, "faiss_index")

    print(f"--- INSPECTION DE L'INDEX FAISS : {index_path} ---\n")

    if not os.path.exists(index_path):
        print("ERREUR : L'index n'existe pas. Lancez d'abord src/indexer.py")
        return

    try:
        # Chargement de la base vectorielle
        vector_store = load_index(index_path)
        
        # Accès au stockage des documents (Docstore)
        # FAISS stocke les docs dans un dictionnaire interne accessible via .docstore._dict
        docs = vector_store.docstore._dict
        
        print(f"Nombre total de chunks (vectors) dans la base : {len(docs)}")
        print("="*60)
        
        # Récupération du premier document pour l'exemple
        first_id = list(docs.keys())[0]
        sample_doc = docs[first_id]
        
        print("EXEMPLE D'UN CHUNK ENREGISTRÉ (ÉCHANTILLON) :")
        print("="*60)
        print(f"ID du vecteur : {first_id}")
        print("-" * 20)
        print("MÉTADONNÉES (Metadata) :")
        for key, value in sample_doc.metadata.items():
            print(f"  - {key}: {value}")
        print("-" * 20)
        print("CONTENU DU TEXTE (page_content) :")
        print(f"'{sample_doc.page_content}'")
        print("="*60)
        print("\nCeci est la preuve de ce qui est réellement stocké dans le fichier index.pkl associé à index.faiss.")

    except Exception as e:
        print(f"Erreur lors de la lecture : {e}")

if __name__ == "__main__":
    inspect_faiss_content()
