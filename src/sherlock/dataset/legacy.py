"""
legacy.py
---------
Import des splits train/val/test du projet de fin d'études (GraduateProject).

Ces splits sont ceux sur lesquels le modèle historique V6 a obtenu
F1 macro = 0,629 sur test. On ne les re-découpe pas : les garder à l'identique
est ce qui rend les résultats comparables entre l'ancien et le nouveau pipeline.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from sherlock.config import cfg

SPLITS = ("train", "val", "test")
REQUIRED_COLS = ["texte", "parti", "media", "sentiment", "ironie"]
SENTIMENTS = {"positif", "neutre", "négatif"}

# Coquilles présentes dans les annotations d'origine
SENTIMENT_FIXES = {"positre": "positif", "negatif": "négatif"}


def _to_bool(series: pd.Series) -> pd.Series:
    """Convertit 'True'/'False'/1/0 en booléens (astype(bool) sur 'False' donne True)."""
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "vrai"})


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise un split : types, coquilles de sentiment, textes vides."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    df = df.copy()
    df["texte"] = df["texte"].astype(str).str.strip()
    df["parti"] = df["parti"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower().replace(SENTIMENT_FIXES)
    df["ironie"] = _to_bool(df["ironie"])

    unknown_parties = set(df["parti"]) - set(cfg.parties)
    if unknown_parties:
        raise ValueError(f"Partis inconnus : {unknown_parties}")

    unknown_sent = set(df["sentiment"]) - SENTIMENTS
    if unknown_sent:
        raise ValueError(f"Sentiments inconnus : {unknown_sent}")

    return df[df["texte"] != ""].reset_index(drop=True)


def load_legacy_splits(src_dir: Path) -> dict[str, pd.DataFrame]:
    """Charge et normalise train/val/test.csv depuis src_dir."""
    splits = {}
    for name in SPLITS:
        path = src_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Split introuvable : {path}")
        df = pd.read_csv(path, sep=cfg.io.separator, encoding=cfg.io.encoding)
        splits[name] = normalize(df)
        logger.info(f"{name} : {len(splits[name]):,} lignes")
    return splits


def overlap_report(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Compte les textes identiques présents dans deux splits (fuite potentielle)."""
    texts = {name: set(df["texte"]) for name, df in splits.items()}
    return {
        "train_val": len(texts["train"] & texts["val"]),
        "train_test": len(texts["train"] & texts["test"]),
        "val_test": len(texts["val"] & texts["test"]),
    }


def import_legacy(src_dir: Path, out_dir: Path) -> dict[str, int]:
    """Convertit les splits CSV historiques en parquet dans out_dir."""
    splits = load_legacy_splits(src_dir)

    overlaps = overlap_report(splits)
    if any(overlaps.values()):
        logger.warning(f"Textes partagés entre splits (conservés tels quels) : {overlaps}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    logger.info(f"Splits parquet écrits dans {out_dir}")

    return {name: len(df) for name, df in splits.items()} | {
        f"overlap_{k}": v for k, v in overlaps.items()
    }
