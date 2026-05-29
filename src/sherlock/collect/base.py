from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import pandas as pd
from loguru import logger

from sherlock.config import cfg


class BaseScraper(ABC):
    """
    Interface commune à tous les scrapers.
    Chaque scraper doit implémenter `fetch()`.
    """

    # Colonnes obligatoires dans tout DataFrame produit
    SCHEMA: ClassVar[list[str]] = ["texte", "compte", "parti", "date", "source", "media"]

    def __init__(self, parti: str):
        if parti not in cfg.parties and parti != "aucun":
            raise ValueError(
                f"Parti '{parti}' inconnu. Valeurs acceptées : {[*cfg.parties, 'aucun']}"
            )
        self.parti = parti
        self.logger = logger.bind(scraper=self.__class__.__name__)

    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """
        Récupère les données et retourne un DataFrame
        avec les colonnes définies dans SCHEMA.
        """

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vérifie que le DataFrame respecte le schéma."""
        missing = [c for c in self.SCHEMA if c not in df.columns]
        if missing:
            raise ValueError(f"{self.__class__.__name__} : colonnes manquantes {missing}")
        return df[self.SCHEMA]

    def save(self, df: pd.DataFrame, out_path: Path) -> None:
        """Sauvegarde le DataFrame en CSV."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            out_path,
            sep=cfg.io.separator,
            encoding=cfg.io.encoding,
            index=False,
        )
        self.logger.info(f"Sauvegardé : {out_path} ({len(df):,} lignes)")
