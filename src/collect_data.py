import requests
import pandas as pd
import os
import re
from datetime import datetime, timedelta
import pytz  # Ensure pytz is installed or use zoneinfo for Python 3.9+

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evenements_culturels.csv")

# Paramètres de filtrage
REGION_FILTER = "Île-de-France"
LIMIT = 99

def get_paris_time():
    """Retourne la date/heure actuelle sur le fuseau Paris."""
    try:
        paris_tz = pytz.timezone("Europe/Paris")
        return datetime.now(paris_tz)
    except Exception:
        # Fallback si pytz n'est pas dispo (bien que standard)
        return datetime.now()

def clean_html(raw_html):
    """Nettoie les balises HTML d'une chaîne de caractères."""
    if not isinstance(raw_html, str):
        return ""
    # Regex pour retirer les balises <...>
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    # Décodage des entités HTML basiques si nécessaire (ex: &amp;)
    # Ici on fait simple pour le POC : suppression des sauts de ligne excessifs
    cleantext = cleantext.replace('\n', ' ').replace('\r', '').replace('&amp;', '&').strip()
    # Suppression des espaces multiples
    cleantext = re.sub(' +', ' ', cleantext)
    return cleantext

def fetch_events():
    """
    Interroge l'API OpenAgenda pour récupérer les événements pertinents.
    Retourne une liste de résultats bruts.
    """
    print(f"Récupération des événements pour la région : {REGION_FILTER}...")

    # Calcul de la date limite : On veut les événements FUTURS ou EN COURS
    # Donc lastdate_end >= Aujourd'hui (Paris)
    now_paris = get_paris_time()
    print(f"Date/Heure d'exécution (Paris) : {now_paris.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # On filtre à partir d'aujourd'hui
    date_str = now_paris.strftime("%Y-%m-%d")

    # Filtrage API (date et région)
    # lastdate_end > date_str ensures we don't get finished events
    where_clause = f'lastdate_end > "{date_str}" AND location_region = "{REGION_FILTER}"'

    params = {
        "limit": LIMIT,
        "where": where_clause,
        "order_by": "firstdate_begin ASC" # Trie par date de début (les plus proches d'abord)
    }

    print(f"Clause WHERE utilisée : {where_clause}")
    print(f"Tri utilisé : firstdate_begin ASC")

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        records = data.get("results", [])
        print(f"{len(records)} événements trouvés via l'API.")
        return records

    except requests.exceptions.HTTPError as err:
        print(f"Erreur HTTP : {err}")
        try:
            print(f"Détail API : {response.text}")
        except:
            pass
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
        # Sélection du titre
        raw_title = item.get("title_fr") or item.get("title_en") or item.get("title") or "Sans titre"
        titre = clean_html(raw_title)

        # Gestion des descriptions
        desc_long = item.get("longdescription_fr") or item.get("longdescription") or ""
        desc_short = item.get("description_fr") or item.get("description") or ""
        
        # Choix de la description la plus riche
        raw_desc = desc_long if len(desc_long) > len(desc_short) else desc_short
        
        # NETTOYAGE HTML CRITIQUE
        description = clean_html(raw_desc)

        if not description or len(description) < 10:
            # Enrichissement par mots-clés si description vide ou trop courte
            keywords = item.get("keywords_fr") or []
            if keywords:
                description = f"Mots-clés : {', '.join(keywords)}"
            else:
                continue 

        lieu = clean_html(item.get("location_name", "Lieu inconnu"))
        date_range = clean_html(item.get("daterange_fr", "Date inconnue"))

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
        print("Aperçu des données nettoyées :")
        print(df[["title", "location"]].head())
    else:
        print("Aucune donnée pertinente n'a pu être traitée.")

if __name__ == "__main__":
    events = fetch_events()
    process_and_save(events)
