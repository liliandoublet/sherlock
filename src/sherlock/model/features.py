"""
features.py
-----------
Injection du sentiment et de l'ironie dans le texte d'entrée.

Reproduit exactement le format du modèle historique V6 :
    "[NOIRONY] [SENT_négatif] texte original"
L'ancien projet n'a pas d'ablation fiable (le modèle « V4 sans métadonnées » a en fait
été évalué avec) : l'ablation est `sherlock train --no-meta`.
"""

import pandas as pd


def meta_prefix(sentiment: str, irony: bool) -> str:
    """Construit le préfixe de métadonnées pour un texte."""
    irony_tag = "[IRONY]" if irony else "[NOIRONY]"
    return f"{irony_tag} [SENT_{sentiment.strip().lower()}] "


def build_inputs(df: pd.DataFrame, use_meta: bool) -> list[str]:
    """Retourne les textes à tokeniser, préfixés ou non par les métadonnées."""
    if not use_meta:
        return df["texte"].tolist()
    return [
        meta_prefix(sent, bool(irony)) + text
        for text, sent, irony in zip(df["texte"], df["sentiment"], df["ironie"], strict=True)
    ]
