"""
Test de diagnostic de la chaîne complète, sans interface graphique.

Utile pour vérifier rapidement, en ligne de commande, que tout fonctionne :
  - sur cette machine après installation,
  - ou après avoir porté le projet sur un autre laptop.

Usage :
  python test_pipeline.py [chemin_audio]

Si aucun fichier audio n'est fourni, utilise test_audio.wav.
"""

import sys
import time
from pathlib import Path

from transcriber import load_model, transcribe_audio
from structure_report import structure_report, check_ollama


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "test_audio.wav"
    if not Path(audio_path).exists():
        print(f"Fichier audio introuvable : {audio_path}")
        sys.exit(1)

    # --- Étape 1 : transcription (CPU) ---
    print("Chargement du modèle Whisper (CPU)...")
    t0 = time.time()
    load_model()
    print(f"  Modèle chargé en {time.time() - t0:.1f} s")

    print(f"\nTranscription de : {audio_path}")
    t0 = time.time()
    text, info = transcribe_audio(audio_path)
    elapsed = time.time() - t0
    facteur = elapsed / info.duration if info.duration else 0

    print("\n" + "=" * 70)
    print("TRANSCRIPTION BRUTE :")
    print("=" * 70)
    print(text)
    print("-" * 70)
    print(
        f"Audio : {info.duration:.1f} s | Transcrit en : {elapsed:.1f} s "
        f"({facteur:.2f}x le temps réel)"
    )

    # --- Étape 2 : structuration (LLM local) ---
    print("\nVérification d'Ollama...")
    ok, msg = check_ollama()
    if not ok:
        print(f"  Structuration ignorée : {msg}")
        return

    print("  OK. Structuration en cours...")
    t0 = time.time()
    report = structure_report(text)
    print("\n" + "=" * 70)
    print("COMPTE-RENDU STRUCTURÉ :")
    print("=" * 70)
    print(report)
    print("-" * 70)
    print(f"Structuré en {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
