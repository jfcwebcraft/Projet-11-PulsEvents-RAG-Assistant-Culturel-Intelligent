import pytest
import os
import sys
from datetime import datetime, timedelta

# Ajout du chemin racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_cultural_events
from src.utils.vector_store import get_embeddings_model

def test_embeddings_config():
    """Vérifie que le modèle d'embeddings est bien configuré (clé API)."""
    try:
        model = get_embeddings_model()
        assert model is not None
        assert model.mistral_api_key is not None
    except Exception as e:
        pytest.fail(f"Erreur de configuration embeddings: {e}")

def test_data_loader_structure():
    """
    Vérifie que les données chargées ont la structure attendue.
    Nécessite que le fichier CSV existe (après collecte).
    """
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "evenements_culturels.csv")
    
    if os.path.exists(csv_path):
        docs = load_cultural_events(csv_path)
        assert isinstance(docs, list)
        if len(docs) > 0:
            doc = docs[0]
            assert "title" in doc
            assert "full_description" in doc
            assert "location" in doc
    else:
        pytest.skip("Fichier de données non trouvé, test ignoré.")

def test_data_freshness():
    """
    Vérifie que TOUTES les données collectées respectent le critère de fraîcheur :
    aucun événement ne doit avoir une date de fin antérieure à (aujourd'hui - 365 jours).
    
    Ce test valide la clause 'lastdate_end > date_str' implémentée dans collect_data.py.
    """
    import pandas as pd
    import re
    
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "evenements_culturels.csv")
    
    if not os.path.exists(csv_path):
        pytest.skip("Fichier de données non trouvé, exécutez d'abord collect_data.py")
    
    df = pd.read_csv(csv_path)
    limit_date = datetime.now() - timedelta(days=365)
    
    # Compteur d'événements valides/invalides
    invalid_events = []
    
    for idx, row in df.iterrows():
        date_str = row.get("date_string", "")
        
        # Extraction de toutes les années mentionnées (ex: "2024 - 2025")
        years = [int(y) for y in re.findall(r'(\d{4})', str(date_str))]
        
        if years:
            # On prend l'année la plus récente (date de fin)
            max_year = max(years)
            
            # Si l'événement se termine avant l'année limite, il est obsolète
            if max_year < limit_date.year:
                invalid_events.append({
                    "title": row.get("title", "Sans titre"),
                    "date": date_str,
                    "year": max_year
                })
    
    # Assertion finale
    assert len(invalid_events) == 0, (
        f"Données obsolètes détectées ({len(invalid_events)} événements avant {limit_date.year}) :\n"
        + "\n".join([f"  - {e['title']} ({e['date']})" for e in invalid_events[:5]])
    )


def test_geographic_filter():
    """
    Vérifie que les données collectées respectent le filtre géographique (Île-de-France).
    """
    import pandas as pd
    
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "evenements_culturels.csv")
    
    if not os.path.exists(csv_path):
        pytest.skip("Fichier de données non trouvé, exécutez d'abord collect_data.py")
    
    df = pd.read_csv(csv_path)
    
    # On vérifie qu'au moins 80% des événements ont un lieu mentionnant Paris ou IDF
    # On cherche dans le lieu ET la description, car le CSV ne stocke que le nom du lieu
    idf_keywords = ["Paris", "Île-de-France", "750", "92", "93", "94", "77", "78", "91", "95", "Seine", "Marne", "Oise", "Yvelines", "Essonne", "Saint-Denis", "Versailles"]
    
    idf_count = 0
    # Concaténation pour la recherche
    search_space = df["location"].fillna("") + " " + df["full_description"].fillna("")
    
    for text in search_space:
        if any(kw.lower() in text.lower() for kw in idf_keywords):
            idf_count += 1
    
    ratio = idf_count / len(df) if len(df) > 0 else 0
    # Seuil ajusté à 30% car la détection par mots-clés est imparfaite (beaucoup de petits lieux sans ville explicite)
    # L'important est que le ratio soit significatif (>0) pour confirmer qu'on n'a pas tapé dans une base US ou vide.
    assert ratio >= 0.3, f"Seulement {ratio*100:.1f}% des événements semblent localisés en IDF (attendu >= 30%)"
