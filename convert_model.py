"""
Conversion du modèle Whisper (medium standard) vers le format CTranslate2.

À LANCER UNE SEULE FOIS (puis à chaque changement de modèle d'origine).

Pourquoi cette étape ?
  Le modèle Hugging Face est au format PyTorch, lent sur CPU. On le convertit
  au format CTranslate2 avec une quantization int8 : le résultat est ~3 à 5 fois
  plus rapide sur CPU et beaucoup plus léger en mémoire.

Le modèle exact (openai/whisper-medium) est défini dans config.py.

Usage :
  python convert_model.py
"""

import sys
from pathlib import Path

from config import WHISPER_HF_MODEL, WHISPER_CT2_DIR


def main():
    out_dir = Path(WHISPER_CT2_DIR)

    # Si le modèle est déjà converti, on ne refait pas le travail.
    if (out_dir / "model.bin").exists():
        print(f"Le modèle converti existe déjà dans : {out_dir}")
        print("Rien à faire. (Supprime ce dossier pour forcer une reconversion.)")
        return

    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # Import tardif : ces librairies ne sont nécessaires que pour la conversion,
    # pas pour l'utilisation quotidienne de l'app.
    from ctranslate2.converters import TransformersConverter
    from transformers import WhisperProcessor

    print(f"Modèle source     : {WHISPER_HF_MODEL}")
    print(f"Dossier de sortie : {out_dir}")
    print()
    print("1/2 - Téléchargement + conversion du modèle (int8)...")
    print("      La première fois, ceci télécharge ~3 Go. Sois patient.")

    converter = TransformersConverter(WHISPER_HF_MODEL)
    converter.convert(str(out_dir), quantization="int8", force=True)

    # faster-whisper a besoin du tokenizer (tokenizer.json) et du préprocesseur
    # audio (preprocessor_config.json) à côté du modèle. On les sauvegarde ici.
    print("2/2 - Sauvegarde du tokenizer et du préprocesseur audio...")
    processor = WhisperProcessor.from_pretrained(WHISPER_HF_MODEL)
    processor.tokenizer.save_pretrained(str(out_dir))
    processor.feature_extractor.save_pretrained(str(out_dir))

    print()
    print(f"Terminé. Modèle prêt dans : {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"\nERREUR pendant la conversion : {exc}", file=sys.stderr)
        sys.exit(1)
