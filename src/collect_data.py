import requests
import pandas as pd
import os
from datetime import datetime, timedelta

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evenements_culturels.csv")

# Paramètres de filtrage
# L'orthographe doit correspondre exactement aux données de l'API (ex: "Île-de-France").
REGION_FILTER = "Île-de-France"
# On limite à 99 car 100 plante parfois.
LIMIT = 99

def fetch_events():
    """
    Interroge l'API OpenAgenda pour récupérer les événements pertinents.
    Retourne une liste de résultats bruts.
    """
    print(f"Récupération des événements pour la région : {REGION_FILTER}...")

    # Calcul de la date limite (-365 jours)
    today = datetime.now()
    one_year_ago = today - timedelta(days=365)
    date_str = one_year_ago.strftime("%Y-%m-%d")

    # Filtrage API (date et région)
    where_clause = f'lastdate_end > "{date_str}" AND location_region = "{REGION_FILTER}"'

    params = {
        "limit": LIMIT,
        "where": where_clause
    }

    print(f"Clause WHERE utilisée : {where_clause}")

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        records = data.get("results", [])
        print(f"{len(records)} événements trouvés via l'API.")
        return records

    except requests.exceptions.HTTPError as err:
        print(f"Erreur HTTP : {err}")
        print(f"Détail API : {response.text}")
        return []
    except Exception as e:
        print(f"Erreur générale : {e}")
        return []

def process_and_save(records):
    if not records:
        print("Aucun enregistrement à sauvegarder.")
        return

    processed_data = []
    for item in records:
        # Sélection du titre (priorité FR > EN > fallback)
        titre = item.get("title_fr") or item.get("title_en") or item.get("title") or "Sans titre"

        # Gestion des descriptions
        desc_long = item.get("longdescription_fr") or item.get("longdescription") or ""
        desc_short = item.get("description_fr") or item.get("description") or ""
        # On maximise le contexte pour le RAG en choisissant la description la plus riche disponible.
        description = desc_long if len(desc_long) > len(desc_short) else desc_short

        if not description:
            # Enrichissement par mots-clés si description vide
            keywords = item.get("keywords_fr") or []
            if keywords:
                description = f"Mots-clés : {', '.join(keywords)}"
            else:
                # Donnée trop pauvre pour être utile au modèle -> rejet
                continue 

        lieu = item.get("location_name", "Lieu inconnu")
        date_range = item.get("daterange_fr", "Date inconnue")

        # Formatage pour le RAG
        full_text = f"Titre: {titre}\nDate: {date_range}\nLieu: {lieu}\nDescription: {description}"

        processed_data.append({
            "uid": item.get("uid"),
            "title": titre,
            "date_string": date_range,
            "location": lieu,
            "full_description": full_text,
            "image_url": item.get("image")
        })

    if processed_data:
        df = pd.DataFrame(processed_data)
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"{len(df)} événements sauvegardés dans : {OUTPUT_FILE}")
        print("Aperçu des données :")
        print(df[["title", "location"]].head())
    else:
        print("Aucune donnée pertinente n'a pu être traitée.")

if __name__ == "__main__":
    events = fetch_events()
    process_and_save(events)
