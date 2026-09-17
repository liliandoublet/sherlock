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


def load_model(model_dir: Path):
    """Charge un modèle fine-tuné et son tokenizer depuis un dossier save_pretrained."""
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"Pas de config.json dans {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    logger.info(f"Modèle chargé : {model_dir}")
    return model, tokenizer
