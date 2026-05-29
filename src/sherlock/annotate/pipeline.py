from pathlib import Path

import pandas as pd
from loguru import logger

from sherlock.annotate.irony import annotate_irony
from sherlock.annotate.sentiment import annotate_sentiment
from sherlock.config import cfg


def annotate(
    df: pd.DataFrame,
    text_col: str = "texte",
    batch_size: int | None = None,
) -> pd.DataFrame:
    """
    Applique sentiment + ironie sur un DataFrame.

    Args:
        df:         DataFrame avec une colonne texte
        text_col:   nom de la colonne texte
        batch_size: taille des batchs

    Returns:
        DataFrame enrichi avec 4 nouvelles colonnes :
        sentiment, sentiment_score, ironie, ironie_score
    """
    logger.info("Pipeline d'annotation démarré")

    df = annotate_sentiment(df, text_col=text_col, batch_size=batch_size)
    df = annotate_irony(df, text_col=text_col, batch_size=batch_size)

    logger.info("Pipeline d'annotation terminé")
    return df


def save_annotated(df: pd.DataFrame, out_path: Path) -> None:
    """Sauvegarde le DataFrame annoté en CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        out_path,
        sep=cfg.io.separator,
        encoding=cfg.io.encoding,
        index=False,
    )
    logger.info(f"Dataset annoté sauvegardé : {out_path} ({len(df):,} lignes)")
