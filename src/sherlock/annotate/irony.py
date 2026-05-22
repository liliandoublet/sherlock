import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm
from loguru import logger

from sherlock.config import cfg


MODEL_NAME = "mrm8488/camembert-base-finetuned-irony-detection"


def annotate_irony(
    df: pd.DataFrame,
    text_col: str = "texte",
    batch_size: int | None = None,
) -> pd.DataFrame:
    """
    Ajoute deux colonnes au DataFrame :
    - ironie       : 'ironic' ou 'not_ironic'
    - ironie_score : score de confiance (0.0 à 1.0)

    Args:
        df:         DataFrame avec une colonne texte
        text_col:   nom de la colonne texte
        batch_size: taille des batchs

    Returns:
        DataFrame avec colonnes ironie et ironie_score ajoutées.
    """
    batch_size = batch_size or cfg.model.batch_size

    if text_col not in df.columns:
        raise ValueError(f"Colonne '{text_col}' absente du DataFrame.")

    logger.info(f"Chargement du modèle ironie : {MODEL_NAME}")
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        device_map="auto",
        truncation=True,
        max_length=cfg.model.max_length,
    )

    texts = df[text_col].fillna("").astype(str).tolist()
    labels, scores = [], []

    logger.info(f"Annotation de {len(texts):,} textes (batch_size={batch_size})")
    for i in tqdm(range(0, len(texts), batch_size), desc="Ironie"):
        batch = texts[i : i + batch_size]
        results = classifier(batch)
        labels.extend(r["label"] for r in results)
        scores.extend(round(r["score"], 4) for r in results)

    df = df.copy()
    df["ironie"]       = labels
    df["ironie_score"] = scores

    dist = df["ironie"].value_counts().to_dict()
    logger.info(f"Distribution ironie : {dist}")
    return df