"""
Structuration d'une transcription brute en compte-rendu médical.

On envoie le texte transcrit à un LLM (modèle de langage) qui tourne EN LOCAL
via Ollama. Aucune donnée ne quitte la machine : c'est essentiel pour le
respect du secret médical et du RGPD.

Le rôle du modèle est volontairement limité : il REFORMATE le texte en sections,
il n'a PAS le droit d'inventer ou de déduire la moindre information médicale.
"""

import requests

from config import OLLAMA_MODEL, OLLAMA_URL


# Consigne donnée au modèle. Le ton est strict pour éviter toute « hallucination »
# (le fait qu'un modèle invente des informations plausibles mais fausses).
SYSTEM_PROMPT = """Tu es un assistant qui met en forme des comptes-rendus d'entretiens psychologiques ou psychiatriques dictés.

RÈGLE ABSOLUE ET NON NÉGOCIABLE :
- Tu ne dois JAMAIS inventer, ajouter, déduire ou compléter une information médicale.
- Tu utilises UNIQUEMENT les informations présentes dans le texte fourni.
- Si une section n'a pas d'information dans le texte, tu ne l'écris pas du tout :
  pas de section vide, jamais de « non renseigné », jamais de « néant ».

CE QUE TU FAIS :
- Tu réorganises le texte dicté en sections claires.
- Tu corriges la ponctuation, l'orthographe et les termes médicaux ou
  psychiatriques manifestement mal transcrits par la reconnaissance vocale
  (mots déformés phonétiquement), en t'appuyant sur le contexte clinique.
  Exemples : « bruchite » → « bronchite », « apirétique » → « apyrétique »,
  « ansiété » → « anxiété », « idénoire » → « idées noires ».
- Corriger la FORME d'un mot mal transcrit n'est PAS inventer : c'est autorisé
  et attendu. En revanche, tu n'ajoutes JAMAIS une information clinique
  (symptôme, diagnostic, traitement, dosage) qui n'est pas dans le texte.

Range les informations dans les sections suivantes, dans cet ordre. N'inclus
une section QUE si le texte contient une information correspondante.

- « Motif de consultation » : la demande du patient, le motif initial de l'entretien.
- « Anamnèse / Histoire » : ce que le patient rapporte lui-même — symptômes ressentis, plaintes, contexte de vie ou professionnel, ancienneté et évolution des troubles.
- « Observation clinique / état psychique » : ce que le clinicien constate pendant l'entretien — présentation, qualité du contact et de l'alliance thérapeutique, humeur, discours, idéation suicidaire.
- « Hypothèse diagnostique » : l'évaluation clinique, le tableau ou le diagnostic évoqué.
- « Conduite à tenir / projet thérapeutique » : suivi proposé, fréquence, approche thérapeutique, traitement, réévaluation.

Règle de tri essentielle : ce que le patient DIT ressentir va dans « Anamnèse » ;
ce que le clinicien OBSERVE de lui va dans « Observation clinique ».

FORMAT DE TA RÉPONSE : écris chaque titre de section sur sa propre ligne, suivi
du contenu rédigé en dessous. NE reproduis PAS les explications ci-dessus :
ta réponse ne contient QUE le compte-rendu lui-même, rien d'autre."""


def check_ollama(model: str | None = None) -> tuple[bool, str]:
    """Vérifie qu'Ollama tourne et que le modèle demandé est disponible.

    Renvoie (True, "") si tout va bien, sinon (False, message d'erreur lisible).
    """
    model = model or OLLAMA_MODEL
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException:
        return (
            False,
            "Ollama ne répond pas. Vérifie qu'il est bien lancé "
            "(l'application Ollama doit tourner en arrière-plan).",
        )

    # On regarde si le modèle demandé fait partie des modèles installés.
    installed = [m.get("name", "") for m in resp.json().get("models", [])]
    if not any(name == model or name.startswith(model) for name in installed):
        return (
            False,
            f"Le modèle « {model} » n'est pas installé dans Ollama.\n"
            f"Installe-le avec la commande : ollama pull {model}",
        )

    return (True, "")


def structure_report(raw_text: str, model: str | None = None) -> str:
    """Transforme une transcription brute en compte-rendu structuré.

    raw_text : le texte issu de la transcription Whisper.
    model    : modèle Ollama à utiliser (par défaut celui de config.py).
    """
    model = model or OLLAMA_MODEL

    user_message = (
        "Voici la transcription brute d'un compte-rendu dicté. "
        "Mets-la en forme selon tes règles :\n\n" + raw_text
    )

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            # Température basse = réponse fidèle et déterministe, peu « créative ».
            "options": {"temperature": 0.1},
        },
        # La génération sur CPU peut être lente sur un laptop : on laisse du temps.
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()
