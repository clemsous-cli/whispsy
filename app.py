"""
Application web locale de transcription médicale.

Lance une petite interface dans le navigateur permettant de :
  1. Importer un fichier audio OU enregistrer au micro.
  2. Obtenir la transcription brute (modèle Whisper médical, en local sur CPU).
  3. Optionnellement, structurer cette transcription en compte-rendu
     (modèle LLM local via Ollama).

Tout fonctionne hors-ligne, en local. Aucune donnée ne quitte la machine.

Usage :
  python app.py
"""

import time

import gradio as gr

from config import OLLAMA_MODEL
from transcriber import transcribe_audio
from structure_report import structure_report, check_ollama


def transcribe(audio_path: str) -> tuple[str, str]:
    """Transcrit un fichier audio et renvoie (texte, message d'info)."""
    if not audio_path:
        raise gr.Error("Aucun audio fourni. Importe un fichier ou enregistre au micro.")

    start = time.time()
    try:
        text, info = transcribe_audio(audio_path)
    except FileNotFoundError as exc:
        # Modèle non converti : on traduit l'erreur en message affiché à l'écran.
        raise gr.Error(str(exc))
    elapsed = time.time() - start

    # On affiche un petit récap de performance (utile pour comparer les machines).
    facteur = elapsed / info.duration if info.duration else 0
    message = (
        f"Transcrit en {elapsed:.1f} s "
        f"(audio : {info.duration:.1f} s, soit {facteur:.1f}x le temps réel)."
    )
    return text, message


def make_report(raw_text: str) -> str:
    """Transforme la transcription brute en compte-rendu structuré (LLM local)."""
    if not raw_text or not raw_text.strip():
        raise gr.Error("Rien à structurer : transcris d'abord un audio.")

    # On vérifie qu'Ollama tourne et que le modèle est installé avant d'envoyer.
    ok, error_message = check_ollama()
    if not ok:
        raise gr.Error(error_message)

    return structure_report(raw_text)


# ---------------------------------------------------------------------------
# Construction de l'interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Transcription médicale (local)") as demo:
    gr.Markdown(
        "# 🩺 Transcription médicale\n"
        "Transcription de comptes-rendus dictés, **100 % en local** sur votre machine. "
        "Aucune donnée n'est envoyée sur Internet."
    )

    with gr.Row():
        # ----- Colonne de gauche : entrée audio + transcription -----
        with gr.Column():
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="1. Audio (importez un fichier ou enregistrez au micro)",
            )
            transcribe_btn = gr.Button("Transcrire", variant="primary")
            perf_info = gr.Markdown("")
            raw_text = gr.Textbox(
                label="2. Transcription brute (modifiable avant structuration)",
                lines=12,
                placeholder="La transcription apparaîtra ici…",
            )

        # ----- Colonne de droite : compte-rendu structuré -----
        with gr.Column():
            structure_btn = gr.Button(
                f"Structurer en compte-rendu (modèle local : {OLLAMA_MODEL})"
            )
            report_text = gr.Textbox(
                label="3. Compte-rendu structuré",
                lines=18,
                placeholder="Le compte-rendu structuré apparaîtra ici…",
            )

    # ----- Câblage des boutons aux fonctions -----
    transcribe_btn.click(
        fn=transcribe,
        inputs=audio_input,
        outputs=[raw_text, perf_info],
    )
    structure_btn.click(
        fn=make_report,
        inputs=raw_text,
        outputs=report_text,
    )


if __name__ == "__main__":
    # inbrowser=True ouvre automatiquement le navigateur sur l'interface.
    demo.launch(inbrowser=True)
