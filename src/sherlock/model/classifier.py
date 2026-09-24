"""
classifier.py
-------------
CamemBERT avec tête de classification (8 partis).

On utilise AutoModelForSequenceClassification : même format de sauvegarde
que les modèles historiques (save_pretrained), donc un seul chemin de code
pour évaluer/servir l'ancien V6 et les nouveaux runs.
"""

from pathlib import Path

from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sherlock.config import cfg


def label_maps(parties: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    """id2label / label2id triés (même ordre que sklearn.LabelEncoder)."""
    labels = sorted(parties)
    return dict(enumerate(labels)), {label: i for i, label in enumerate(labels)}


def build_model(parties: list[str] | None = None, model_name: str | None = None):
    """Instancie CamemBERT pré-entraîné avec une tête de classification neuve."""
    parties = parties or cfg.parties
    model_name = model_name or cfg.model.name
    id2label, label2id = label_maps(parties)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    logger.info(f"Modèle initialisé : {model_name} -> {len(id2label)} classes")
    return model


def load_model(model_dir: str | Path):
    """
    Charge un modèle fine-tuné et son tokenizer.

    `model_dir` est soit un dossier local (format save_pretrained), soit un identifiant
    Hugging Face Hub (« utilisateur/dépôt »), téléchargé puis mis en cache au premier appel.
    """
    source = str(model_dir)
    local = Path(source)
    if local.exists() and not (local / "config.json").exists():
        raise FileNotFoundError(f"Pas de config.json dans {local}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source)
    except (OSError, ValueError) as error:
        raise FileNotFoundError(
            f"Modèle introuvable : « {source} » n'est ni un dossier local ni un dépôt "
            "Hugging Face accessible."
        ) from error
    logger.info(f"Modèle chargé : {source}")
    return model, tokenizer
