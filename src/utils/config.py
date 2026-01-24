import os
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
# On remonte d'un niveau (src/utils -> src) puis d'un autre (src -> root) pour trouver .env
# Ou on cherche explicitement à la racine du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    print("ATTENTION : La clé API MISTRAL_API_KEY est introuvable dans les variables d'environnement.")
