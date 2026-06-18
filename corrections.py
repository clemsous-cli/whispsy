"""
Dictionnaire de corrections déterministes de la transcription.

Principe : on remplace des erreurs de transcription connues par leur forme
correcte, de manière fiable et SANS aucun risque d'invention (contrairement à
un modèle d'IA). Cette étape ne corrige que les erreurs listées ici.

C'est volontairement simple et fait pour être enrichi : quand vous repérez une
erreur récurrente dans vos transcriptions, ajoutez-la au dictionnaire ci-dessous.
"""

import re

# Clé   = ce que Whisper produit PAR ERREUR
# Valeur = la forme correcte attendue
# La correspondance ignore la casse et respecte les limites de mots.
CORRECTIONS = {
    # --- Erreurs médicales observées ---
    "apirétique": "apyrétique",
    "osculitation": "auscultation",
    "osculation": "auscultation",
    "bruchite": "bronchite",
    "paracétamole": "paracétamol",
    "paquets à nez": "paquets-année",

    # --- Ajoutez ici vos corrections psychologie / psychiatrie ---
    # "erreur fréquente": "forme correcte",
}


def apply_corrections(text: str) -> str:
    """Applique toutes les corrections du dictionnaire au texte transcrit."""
    for wrong, right in CORRECTIONS.items():
        # \b délimite des mots entiers ; re.IGNORECASE matche quelle que soit
        # la casse. re.escape protège les éventuels caractères spéciaux.
        pattern = r"\b" + re.escape(wrong) + r"\b"
        text = re.sub(pattern, right, text, flags=re.IGNORECASE)
    return text
