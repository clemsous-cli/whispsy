"""
Configuration centrale de l'application de transcription médicale.

Tout est pensé pour fonctionner en CPU pur, afin que l'application soit
portable sur des laptops modestes sans carte graphique.
Un seul fichier à modifier pour ajuster le comportement de l'app.
"""

from pathlib import Path

# Dossier racine du projet (le dossier qui contient ce fichier)
BASE_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Modèle de TRANSCRIPTION (Whisper)
# ---------------------------------------------------------------------------

# Identifiant du modèle d'origine sur Hugging Face.
# On utilise le Whisper "medium" STANDARD (non spécialisé) : il transcrit les
# longues dictées sans les tronquer (au contraire des versions fine-tunées
# médicales, qui s'arrêtent au bout de ~40 s) et reste rapide sur CPU.
# La précision sur le jargon est ensuite renforcée par trois couches :
# un prompt d'orientation (ci-dessous), un dictionnaire de corrections
# (corrections.py) et la correction faite par le modèle de compte-rendu.
WHISPER_HF_MODEL = "openai/whisper-medium"

# Dossier où est stocké le modèle converti au format CTranslate2 (rapide sur CPU).
WHISPER_CT2_DIR = BASE_DIR / "models" / "whisper-medium-std-ct2"

# Type de calcul : "int8" applique une quantization qui rend le modèle plus
# petit et plus rapide sur CPU, avec une perte de qualité quasi imperceptible.
WHISPER_COMPUTE_TYPE = "int8"

# Langue forcée. Le modèle est entraîné pour le français, on le lui impose
# pour éviter toute mauvaise détection automatique de langue.
WHISPER_LANGUAGE = "fr"

# Nombre de threads CPU utilisés pour la transcription.
# 0 = laisse CTranslate2 choisir automatiquement selon la machine.
WHISPER_CPU_THREADS = 0

# Largeur du "faisceau" de recherche. Plus élevé = un peu plus précis mais
# plus lent. 5 est le bon compromis. Mettre 1 pour aller plus vite sur un
# laptop très modeste (qualité légèrement moindre).
WHISPER_BEAM_SIZE = 5

# Filtre de détection de la voix (VAD) : découpe l'audio aux silences et ne
# transcrit que les passages parlés. Plus rapide et réduit les hallucinations
# (texte inventé pendant les silences). Recommandé : True.
WHISPER_VAD_FILTER = True

# Prompt d'orientation donné à Whisper avant la transcription. Il sert à
# "amorcer" le modèle avec le vocabulaire et le style attendus : ici, la
# psychologie et la psychiatrie. Whisper a alors tendance à mieux transcrire
# ces termes. Adaptez ce texte à votre spécialité si besoin.
WHISPER_INITIAL_PROMPT = (
    "Compte-rendu d'entretien psychologique et psychiatrique. "
    "Le patient présente une anxiété, des ruminations, une humeur dépressive, "
    "des troubles du sommeil et de la concentration. Évaluation de l'état psychique, "
    "des idées suicidaires et de l'alliance thérapeutique. "
    "Diagnostic : trouble anxieux généralisé, épisode dépressif caractérisé, "
    "trouble bipolaire, trouble de la personnalité. "
    "Prise en charge : psychothérapie, thérapie cognitivo-comportementale, "
    "traitement anxiolytique, antidépresseur, thymorégulateur, neuroleptique."
)


# ---------------------------------------------------------------------------
# Modèle de STRUCTURATION (LLM local, via Ollama)
# ---------------------------------------------------------------------------

# Modèle utilisé pour transformer la transcription brute en compte-rendu
# structuré. Qwen2.5 3B = bon compromis qualité / vitesse sur un laptop 16 Go.
# Pour tester un modèle plus puissant sur une grosse machine, remplace par
# par exemple "qwen2.5:7b-instruct".
OLLAMA_MODEL = "qwen2.5:3b-instruct"

# Adresse du service Ollama qui tourne en local sur la machine.
OLLAMA_URL = "http://localhost:11434"
