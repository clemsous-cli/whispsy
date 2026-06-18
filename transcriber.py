"""
Chargement du modèle Whisper médical et transcription audio (CPU).

Ce module centralise toute la logique de transcription pour qu'elle soit
identique dans l'application (app.py) et dans le script de test (test_pipeline.py).

On utilise le Whisper "medium" standard, qui gère nativement les longues
dictées (pas de troncature). Le réglage condition_on_previous_text=False
limite les répétitions de phrases inventées (hallucinations) en fin d'audio.
"""

from faster_whisper import WhisperModel

from config import (
    WHISPER_CT2_DIR,
    WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE,
    WHISPER_CPU_THREADS,
    WHISPER_BEAM_SIZE,
    WHISPER_VAD_FILTER,
    WHISPER_INITIAL_PROMPT,
)
from corrections import apply_corrections


# Le modèle est chargé une seule fois puis réutilisé (il pèse ~750 Mo en RAM).
_model = None


def load_model() -> WhisperModel:
    """Charge le modèle Whisper la première fois, le réutilise ensuite.

    Lève FileNotFoundError si le modèle n'a pas encore été converti.
    """
    global _model
    if _model is None:
        if not (WHISPER_CT2_DIR / "model.bin").exists():
            raise FileNotFoundError(
                "Le modèle de transcription n'a pas été trouvé.\n"
                "Lance d'abord la conversion : python convert_model.py"
            )
        # device="cpu" en dur : on force le CPU pour rester portable sur tout laptop.
        _model = WhisperModel(
            str(WHISPER_CT2_DIR),
            device="cpu",
            compute_type=WHISPER_COMPUTE_TYPE,
            cpu_threads=WHISPER_CPU_THREADS,
        )
    return _model


def transcribe_audio(audio_path: str):
    """Transcrit un fichier audio et renvoie (texte, info).

    `info` contient notamment la durée de l'audio (info.duration).
    """
    model = load_model()
    segments, info = model.transcribe(
        audio_path,
        language=WHISPER_LANGUAGE,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=WHISPER_VAD_FILTER,
        # Oriente le modèle vers le vocabulaire psy/psychiatrie (voir config.py).
        initial_prompt=WHISPER_INITIAL_PROMPT,
        # N'utilise pas le texte déjà transcrit comme contexte : évite que le
        # modèle se mette à répéter / inventer des phrases en boucle.
        condition_on_previous_text=False,
    )
    # Les segments sont produits "à la demande" : on les assemble en un texte.
    text = " ".join(segment.text.strip() for segment in segments).strip()
    # Dictionnaire de corrections déterministes (coquilles connues).
    text = apply_corrections(text)
    return text, info
