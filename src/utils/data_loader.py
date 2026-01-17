import pandas as pd
import os

def load_cultural_events(csv_path: str):
    """
    Charge le fichier CSV des événements culturels.
    Retourne une liste de documents (dictionnaires) ou un DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Le fichier de données n'existe pas : {csv_path}")
    
    df = pd.read_csv(csv_path)
    # On s'assure que les colonnes nécessaires sont là
    required_cols = ["title", "full_description", "location", "date_string", "uid"]
    for col in required_cols:
        if col not in df.columns:
            # Fallback ou erreur si critique. 
            # Le script collect_data.py génère ces colonnes.
            print(f"Attention: colonne manquante {col}")
    
    # Conversion en liste de dictionnaires pour faciliter le traitement
    documents = df.to_dict(orient="records")
    return documents
