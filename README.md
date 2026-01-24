# 🎭 Puls-Events RAG - Assistant Culturel Intelligent

Ce projet est un **Proof of Concept (POC)** réalisé dans le cadre de la formation Data Engineering. Il déploie un système **RAG (Retrieval-Augmented Generation)** permettant aux utilisateurs d'interagir en langage naturel pour découvrir des événements culturels en Île-de-France.

## 🏗 Architecture Technique

Le pipeline de données suit une architecture modulaire :

1.  **Ingestion** (`src/collect_data.py`) : Collecte des événements culturels récents via l'API publique OpenAgenda.
2.  **Vectorisation** (`src/indexer.py`) : 
    *   Enrichissement sémantique des données.
    *   Encodage via **Mistral AI Embeddings**.
    *   Indexation vectorielle avec **Facebook AI Similarity Search (FAISS)**.
3.  **Moteur RAG** (`src/rag_engine.py`) : Encapsule la logique de récupération et de génération, incluant les garde-fous (hallucinations, hors-sujet).
4.  **Application** (`src/app.py`) : Interface **Streamlit** intégrant le modèle **Mistral-Large** pour générer des recommandations personnalisées basées sur le contexte vectoriel.

## 🚀 Installation et Démarrage

### Prérequis
*   Python 3.10+
*   Une clé API Mistral (`MISTRAL_API_KEY`)

### 1. Configuration de l'environnement

```powershell
# Créer l'environnement virtuel
python -m venv venv

# Activation (Windows PowerShell) :
.\venv\Scripts\Activate.ps1
```

> **Note importante** : Sous Windows, il est fréquent que PowerShell utilise le Python global malgré l'activation. Pour garantir l'utilisation des dépendances du projet, nous utiliserons le chemin absolu vers l'exécutable Python du venv dans les commandes ci-dessous.

```powershell
# Installer les dépendances (via le pip du venv)
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Configuration des secrets (informations sensibles)
Créez un fichier `.env` à la racine (ne pas commiter ce fichier) :
```env
MISTRAL_API_KEY=votre_clé_api_ici
```

### 3. Pipeline de Données
Initialisez la base de connaissances en exécutant les scripts ETL :

```powershell
# 1. Collecte (API OpenAgenda -> RAW CSV)
.\venv\Scripts\python.exe src/collect_data.py

# 2. Indexation (CSV -> Embeddings -> FAISS)
.\venv\Scripts\python.exe src/indexer.py
```

### 4. Lancer l'Assistant
Démarrez l'interface Streamlit :

```powershell
.\venv\Scripts\python.exe -m streamlit run src/app.py
```

## 🛠 Choix Techniques

*   **Mistral AI** : Choisi pour sa performance supérieure en langue française et son ratio coût/efficacité.
*   **FAISS (CPU)** : Librairie standard de l'industrie pour la recherche de similarité, suffisante pour le volume de données de ce POC (< 10k items).
*   **LangChain** : Framework pour orchestrer proprement la chaîne RAG (Retriever + Prompt + LLM).
*   **Streamlit** : Permet un prototypage rapide d'interface utilisateur data-driven.

## 📂 Structure du Projet

*   `data/` : Données brutes et procesées (non versionnées).
*   `faiss_index/` : Base de données vectorielle locale.
*   `src/` : Code source des composants (Collecte, Indexation, Moteur RAG, App).
*   `tests/` : Suite de tests automatisés.

## ✅ Tests et Validation

Le projet inclut une suite de tests complète :

*   **Tests d'Infrastructure** (`tests/test_rag.py`) : Valident le chargement des données, les filtres géographiques et la configuration des embeddings.
*   **Tests d'Évaluation QA** (`tests/test_evaluation.py`) : Valident la qualité des réponses (fidelité au contexte, refus des hallucinations, robustesse aux prompts adverses).
    *   Basé sur le dataset : `tests/test_dataset_qa.csv`.

Pour lancer les tests :
```powershell
.\venv\Scripts\python.exe -m pytest tests/
```
