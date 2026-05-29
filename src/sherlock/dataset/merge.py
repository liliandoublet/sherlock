from pathlib import Path

import pandas as pd
from loguru import logger

from sherlock.config import cfg

# Colonnes obligatoires dans chaque source
REQUIRED_COLS = ["texte", "compte", "parti", "date", "source", "media"]

# Corrections de labels connus
LABEL_FIXES = {
    "Reconquete": "Reconquête",
    "Reconquête": "Reconquête",
    "LFI_communiques": "LFI",
}


def _load_one(path: Path) -> pd.DataFrame:
    """Charge un CSV, vérifie les colonnes, normalise."""
    df = pd.read_csv(
        path,
        sep=cfg.io.separator,
        encoding=cfg.io.encoding,
        dtype=str,
    )

    # Renomme 'média' -> 'media' si besoin (bug Wikipedia)
    if "média" in df.columns and "media" not in df.columns:
        df = df.rename(columns={"média": "media"})

    # Supprime chunk_idx si présent (héritage du pipeline web)
    if "chunk_idx" in df.columns:
        df = df.drop(columns=["chunk_idx"])

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} : colonnes manquantes {missing}")

    return df[REQUIRED_COLS]


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige les variantes de labels de partis."""
    df["parti"] = df["parti"].replace(LABEL_FIXES)
    return df


def merge_sources(*paths: Path) -> pd.DataFrame:
    """
    Charge et fusionne plusieurs CSV sources.
    Normalise les labels et retire les doublons sur 'texte'.

    Args:
        *paths: chemins vers les CSV à fusionner

    Returns:
        DataFrame fusionné et nettoyé.
    """
    frames = []
    for p in paths:
        logger.info(f"Chargement : {p.name}")
        df = _load_one(p)
        frames.append(df)
        logger.debug(f"  {len(df):,} lignes")

    merged = pd.concat(frames, ignore_index=True)
    logger.info(f"Après fusion : {len(merged):,} lignes")

    merged = _normalize_labels(merged)

    before = len(merged)
    merged = merged.drop_duplicates(subset=["texte"]).reset_index(drop=True)
    logger.info(
        f"Après déduplication : {len(merged):,} lignes ({before - len(merged):,} doublons supprimés)"
    )

    return merged
