import pandas as pd
from loguru import logger

from sherlock.config import cfg
from sherlock.clean.regex_rules import clean_text, count_words_excluding_token
from sherlock.clean.language import is_french


def run(df: pd.DataFrame, source: str = "twitter") -> pd.DataFrame:
    """
    Pipeline complet de nettoyage.

    Étapes :
      1. Supprime les retweets (Twitter uniquement)
      2. Filtre sur le français
      3. Nettoie le texte (regex)
      4. Filtre les textes trop courts
      5. Déduplique

    Args:
        df:      DataFrame avec au minimum une colonne 'texte'
        source:  'twitter' ou 'web' (adapte certaines étapes)

    Returns:
        DataFrame nettoyé avec colonne 'texte' remplacée par le texte propre.
    """
    initial = len(df)
    logger.info(f"Nettoyage démarré : {initial:,} lignes ({source})")

    # 1. Supprimer les retweets (Twitter seulement)
    if source == "twitter":
        df = df[~df["texte"].str.startswith("rt @", na=False)]
        logger.debug(f"Après suppression RT : {len(df):,} lignes")

    # 2. Garder uniquement le français
    df = df[df["texte"].apply(is_french)].copy()
    logger.debug(f"Après filtre FR : {len(df):,} lignes")

    # 3. Nettoyage textuel
    token = cfg.cleaning.mention_token
    df["texte"] = df["texte"].apply(
        lambda t: clean_text(t, mention_token=token)
    )

    # 4. Filtrer les textes trop courts
    min_words = cfg.cleaning.min_words_strict
    df = df[
        df["texte"].apply(
            lambda t: count_words_excluding_token(t, token) >= min_words
        )
    ]
    logger.debug(f"Après filtre longueur (>={min_words} mots) : {len(df):,} lignes")

    # 5. Dédupliquer sur le texte nettoyé
    df = df.drop_duplicates(subset=["texte"]).reset_index(drop=True)

    logger.info(
        f"Nettoyage terminé : {len(df):,}/{initial:,} lignes conservées "
        f"({len(df)/initial*100:.1f}%)"
    )
    return df