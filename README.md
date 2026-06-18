# 🩺 whispsy

**Transcription locale de comptes-rendus dictés, orientée psychologie / psychiatrie.**

Application fonctionnant **100 % en local** sur la machine : aucune donnée audio
ou texte ne quitte l'ordinateur. Conçue pour respecter le **secret médical** et
le **RGPD**.

> ⚕️ **Avertissement** — whispsy est un outil d'aide à la rédaction, **pas un
> dispositif médical certifié**. Les transcriptions et comptes-rendus générés
> doivent toujours être **relus et validés** par le praticien, qui reste seul
> responsable du contenu. Fourni « tel quel », sans aucune garantie.

## Comment ça marche

Le traitement se fait en deux temps, avec **trois couches de correction** qui
améliorent la précision sans jamais inventer de contenu :

```
1. TRANSCRIPTION   Whisper medium (standard)          audio  -> texte brut
   ├─ couche 1 : prompt d'orientation psy/psychiatrie (oriente le vocabulaire)
   └─ couche 2 : dictionnaire de corrections déterministes (coquilles connues)

2. STRUCTURATION   Qwen2.5 3B via Ollama              texte  -> compte-rendu
   └─ couche 3 : correction du jargon mal transcrit, en s'appuyant sur le contexte
```

> **Pourquoi Whisper « standard » et pas un modèle « médical » ?**
> Les modèles Whisper fine-tunés médicaux testés **tronquaient** les dictées de
> plus de ~40 s et faisaient *plus* de fautes. Le Whisper standard gère les
> longues dictées nativement et transcrit très bien le français ; la précision
> sur le jargon est rétablie par les trois couches de correction ci-dessus.

Tout est pensé pour tourner en **CPU pur**, afin d'être portable sur des
laptops modestes (16 Go de RAM suffisent).

---

## Prérequis

- **Python 3.13** (ou 3.12)
- **[Ollama](https://ollama.com/download)** installé (pour la structuration)

---

## Installation (à faire une fois)

### 0. Récupérer le projet

```powershell
git clone https://github.com/clemsous-cli/whispsy.git
cd whispsy
```

### 1. Environnement Python et dépendances

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# torch en version CPU pure (plus légère)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Convertir le modèle de transcription

Télécharge Whisper medium (~3 Go la première fois) et le convertit en format
optimisé CPU. **À ne faire qu'une seule fois.**

```powershell
python convert_model.py
```

Le modèle converti est stocké dans `models/whisper-medium-std-ct2/` (~770 Mo).

### 3. Récupérer le modèle de structuration

```powershell
ollama pull qwen2.5:3b-instruct
```

---

## Utilisation au quotidien

```powershell
.venv\Scripts\Activate.ps1
python app.py
```

L'interface s'ouvre dans le navigateur. On y :

1. Importe un fichier audio **ou** enregistre au micro.
2. Clique sur **Transcrire** → la transcription apparaît (modifiable :
   **relisez-la**, c'est votre filet de sécurité).
3. *(optionnel)* Clique sur **Structurer en compte-rendu**.

> 💡 Pas de micro sur un PC de bureau ? C'est normal (les tours n'en ont pas).
> Branchez un casque-micro, ou importez un fichier audio.

---

## Vérifier que tout fonctionne

```powershell
python test_pipeline.py              # utilise test_audio.wav
python test_pipeline.py mon_audio.wav  # ou votre propre fichier
```

Affiche la transcription, le compte-rendu et la **vitesse réelle** sur la machine.

---

## Personnaliser

| Fichier | À quoi ça sert |
|---------|----------------|
| [`config.py`](config.py) | Tous les réglages : modèle de structuration, langue, et surtout le **prompt d'orientation** (`WHISPER_INITIAL_PROMPT`) à adapter à votre spécialité. |
| [`corrections.py`](corrections.py) | Le **dictionnaire de corrections**. Ajoutez-y les erreurs récurrentes que vous repérez (`erreur` → `forme correcte`). |
| [`structure_report.py`](structure_report.py) | La consigne donnée au modèle de compte-rendu (sections, règles anti-invention). |

Pour tester un modèle de structuration plus puissant (sur une grosse machine) :
dans `config.py`, remplacez `qwen2.5:3b-instruct` par `qwen2.5:7b-instruct`.

---

## Portabilité vers un autre laptop

1. Installer **Python** et **Ollama** sur la nouvelle machine.
2. Copier le dossier du projet (**sans** `.venv`, à recréer localement).
3. Refaire les étapes **1, 2 et 3** de l'installation.

> Le dossier `models/` (modèle converti) **est** copiable tel quel, ce qui
> évite de re-télécharger les 3 Go. Seul le `.venv` doit être recréé.

---

## Confidentialité

- La **transcription** fonctionne entièrement hors-ligne.
- La **structuration** appelle Ollama **en local** (`localhost`) : aucune donnée
  ne sort de la machine, aucune clé API, aucun cloud.

---

## Licence

Distribué sous licence **Apache 2.0** — voir [LICENSE](LICENSE).
Copyright © 2026 clemsous-cli.

Les modèles utilisés ont leurs propres licences (Whisper : MIT ; Qwen2.5 :
Apache 2.0) et **ne sont pas redistribués** dans ce dépôt : ils se téléchargent
à l'installation.
