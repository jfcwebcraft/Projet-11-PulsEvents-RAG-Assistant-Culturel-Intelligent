# Rapport Technique – Assistant Culturel Puls-Events (RAG)

**Projet :** POC – Système de recommandation d'événements via IA générative  
**Date :** Janvier 2026  
**Technologie :** RAG (Retrieval-Augmented Generation)

---

## 1. Contexte et Objectifs

Dans le cadre de la modernisation de la plateforme **Puls-Events**, un assistant culturel intelligent a été développé afin d'améliorer l'expérience utilisateur.
L'enjeu principal consistait à dépasser le système de recherche par mots-clés pour proposer une interface conversationnelle capable de comprendre une intention, comme *« Je cherche une sortie jazz ce week-end »*, et d'y répondre par des recommandations pertinentes et contextualisées.

La solution retenue repose sur un modèle **RAG (Retrieval-Augmented Generation)**, combinant les capacités de génération d'un LLM (Large Language Model) à la fiabilité d'une base de données structurée et mise à jour en continu.

---

## 2. Architecture Technique

L'architecture du système s'articule autour de quatre modules principaux :

1.  **ETL – Ingestion & Nettoyage** : récupération et préparation des données brutes.
2.  **Indexation sémantique** : transformation des textes en vecteurs pour la recherche.
3.  **Moteur RAG** : coordination de la recherche et de la génération de réponses.
4.  **Interface utilisateur** : application web interactive développée avec **Streamlit**.

### Stack Technologique

*   **Langage** : Python 3.10+
*   **LLM et plongements lexicaux** : Mistral AI (`mistral-large-latest`, `mistral-embed`)
*   **Orchestration** : LangChain
*   **Base vectorielle** : FAISS (Facebook AI Similarity Search, mode CPU local)
*   **Source de données** : API OpenAgenda

---

## 3. Pipeline de Données (Pré-traitement)

Un soin particulier a été apporté à la qualité des données en amont du traitement, afin de garantir la pertinence des résultats générés.

### 3.1 Collecte (`src/collect_data.py`)

Les données sont extraites de l'API publique **OpenAgenda**, avec plusieurs filtres destinés à réduire le bruit :

*   **Filtre géographique** : `region = "Île-de-France"` (périmètre du POC).
*   **Filtre temporel** : exclusion des événements dont la date de fin est antérieure à un an.
*   **Sélection des champs** : accent mis sur les titres et descriptions détaillées en français.

### 3.2 Vectorisation (`src/indexer.py`)

Les textes collectés sont convertis en embeddings pour permettre la recherche sémantique :

*   **Enrichissement des documents** : concaténation des champs `Titre + Date + Lieu + Description`.
*   **Chunking** : découpage des contenus longs (taille : 1000 caractères) via `RecursiveCharacterTextSplitter`.
*   **Stockage local** : l'index FAISS est sauvegardé sur disque, garantissant un temps de réponse inférieur à 50 ms sans dépendance réseau.

---

## 4. Le Système RAG (`src/rag_engine.py`)

Le moteur RAG constitue le cœur du projet. Il gère la récupération des documents pertinents et la génération de la réponse finale.

### 4.1 Logique de récupération

Pour chaque question utilisateur :
1.  La requête est vectorisée.
2.  FAISS identifie les trois documents les plus proches sémantiquement (`k = 3`).
3.  Ces documents sont insérés dans le contexte du prompt transmis au modèle.

### 4.2 Prompt Engineering et contrôle qualité

Un ensemble de garde-fous encadre le comportement du modèle pour éviter toute dérive :

*   **Principe de réalité** : interdiction d'inventer des événements inexistants.
*   **Filtrage thématique** : refus automatique des sujets hors culture (politique, humour, etc.).
*   **Précision** : si la demande est trop vague, le modèle reformule pour obtenir des précisions.
*   **Formatage** : les réponses suivent un format défini (Titre, Date, Lieu, Argumentaire).

---

## 5. Validation et Contrôle Qualité

Avant la mise en production, le système a été soumis à une batterie de tests techniques et fonctionnels.

### 5.1 Tests unitaires (`tests/test_rag.py`)

Ces tests vérifient la cohérence et la fiabilité du pipeline :
*   Absence d'événements obsolètes.
*   Respect du périmètre géographique (Île-de-France).
*   Connexion correcte à l'API Mistral et configuration des embeddings.

### 5.2 Tests fonctionnels et métier (`tests/test_evaluation.py`)

Ces tests mettent en évidence la compréhension et la robustesse du modèle à partir de données annotées (`test_dataset_qa.csv`) :
*   **Tests d'hallucination** : vérification que le modèle refuse d'inventer des événements.
*   **Tests adversaires** : contrôle du maintien des garde-fous face aux requêtes de contournement.
*   **Tests de précision** : comparaison entre la réponse générée et les contextes extraits.

### 5.3 Résultats

Les résultats de la phase de test ont confirmé la stabilité du système :
*   **100 %** des tests unitaires validés.
*   **100 %** de réussite sur les tests d'évaluation et de sécurité.

---

## 6. Conclusion et perspectives

Le POC confirme la faisabilité technique et la pertinence d'un assistant culturel fondé sur l'approche RAG.
Les performances enregistrées sont satisfaisantes, les coûts maîtrisés (grâce à l'usage raisonné de l'API Mistral), et les garde-fous permettent de fiabiliser la qualité des recommandations.

**Pistes d'évolution pour la version 1 :**

1.  **Scalabilité** : migration de FAISS local vers une base vectorielle cloud (Pinecone, Qdrant) pour gérer des volumes supérieurs à un million d'événements.
2.  **Actualisation en continu** : automatisation de la mise à jour de l'index via les webhooks OpenAgenda.
3.  **Personnalisation** : exploitation de l'historique utilisateur pour générer des recommandations tenant compte des préférences passées.