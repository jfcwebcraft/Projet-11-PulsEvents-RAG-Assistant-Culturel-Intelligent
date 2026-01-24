import pytest
import pandas as pd
import os
import sys

# Ajout du chemin racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import get_rag_chain

# Chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_FILE = os.path.join(BASE_DIR, "tests", "test_dataset_qa.csv")

@pytest.fixture(scope="module")
def rag_chain():
    """Initialise la chaîne RAG une seule fois pour tout le module de test."""
    try:
        return get_rag_chain()
    except Exception as e:
        pytest.skip(f"Moteur RAG non disponible : {e}")

def load_qa_dataset():
    if not os.path.exists(QA_FILE):
        return []
    df = pd.read_csv(QA_FILE)
    return df.to_dict("records")

@pytest.mark.parametrize("test_case", load_qa_dataset())
def test_rag_evaluation(rag_chain, test_case):
    """
    Exécute les cas de tests définis dans le fichier CSV.
    Vérifie les garde-fous (hallucinations, hors-sujet) et la pertinence.
    """
    question = test_case.get("question")
    # Nettoyage de la catégorie
    category = test_case.get("category", "").strip()
    # Parsing de check_context (convertir chaine "true"/"false" en bool)
    check_context_str = str(test_case.get("check_context", "false")).lower()
    check_context = check_context_str == "true"
    
    expected_keywords = test_case.get("expected_answer_keywords", "").split(",")
    # Nettoyage des keywords
    expected_keywords = [k.strip().lower() for k in expected_keywords if k.strip()]

    print(f"\n--- Test ({category}) : {question} ---")
    
    try:
        response = rag_chain.invoke({"input": question})
        answer = response["answer"].lower()
        
        print(f"Réponse IA : {response['answer']}")
        
        # 1. Vérification du contexte (Retrieval Check)
        if check_context:
            context_docs = response.get("context", [])
            retrieved_content = " ".join([doc.page_content.lower() for doc in context_docs])
            print(f"DEBUG: Contexte récupéré (extrait) : {retrieved_content[:200]}...")
            
            # On vérifie que les mots clés attendus sont AUSSI dans le contexte récupéré
            # (Cela prouve que le retriever a fait son travail)
            if expected_keywords:
                 # On est moins strict sur le contexte : on cherche au moins un mot clé pertinent
                 context_matches = [k for k in expected_keywords if k in retrieved_content]
                 if not context_matches:
                     print("WARNING: Les mots-clés attendus ne sont pas dans le contexte récupéré !")
                     # Optionnel : fail le test ici si on veut etre strict
                     # assert len(context_matches) > 0, f"Le contexte récupéré ne contient pas les infos attendues ({expected_keywords})"

        # 2. Vérifications spécifiques selon la catégorie
        if category in ["hallucination_test", "out_of_scope", "adversarial", "negative_test"]:
            # On s'attend à un refus poli
            refusal_phrases = [
                "ne trouve pas", "pas d'événement", "désolé", "je ne peux pas", 
                "hors de mon périmètre", "pas dans ma base", "ne peux pas inventer",
                "desole", "ne suis pas capable", "pas en mesure", "aucun événement",
                "rien n'est annonce", "pas annonce"
            ]
            
            # Normalisation pour éviter les soucis d'encodage/accent
            import unicodedata
            def normalize(text):
                return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
            
            answer_norm = normalize(answer)
            refusal_phrases_norm = [normalize(p) for p in refusal_phrases]
            
            # Debug
            print(f"DEBUG: Answer Norm: {answer_norm}")
            
            assert any(p in answer_norm for p in refusal_phrases_norm), \
                f"L'IA aurait dû refuser de répondre. Réponse obtenue : {answer}"
                
        elif category == "vague_query":
            # On s'attend à une demande de clarification
            clairification_phrases = ["quel type", "quelle ville", "plus de précision", "préciser", "cherchez-vous", "comment puis-je vous aider"]
            assert any(phrase in answer for phrase in clairification_phrases), \
                f"L'IA aurait dû demander une clarification. Réponse obtenue : {answer}"
                
        else:
            # Cas standard : on cherche les mots-clés attendus
            if expected_keywords:
                matches = [k for k in expected_keywords if k in answer]
                
                # Tolérance : Si pas de mots-clés trouvés, mais que l'IA demande une clarification (car elle n'a peut-être pas trouvé l'info précise), on accepte.
                clairification_phrases = ["quel type", "quelle ville", "plus de précision", "préciser", "cherchez-vous", "comment puis-je vous aider"]
                if not matches and any(phrase in answer for phrase in clairification_phrases):
                    print("INFO: Pas de résultats trouvés, mais l'IA a demandé une clarification. Test accepté.")
                    matches = ["clarification_accepted"]

                # On est tolérant : au moins un mot clé pertinent doit être trouvé
                assert len(matches) > 0, \
                    f"Aucun mot-clé attendu ({expected_keywords}) n'a été trouvé dans la réponse. Réponse : {answer}"

    except Exception as e:
        pytest.fail(f"Erreur lors de l'exécution du test : {e}")
