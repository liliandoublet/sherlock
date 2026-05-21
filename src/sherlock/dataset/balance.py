import pandas as pd
from loguru import logger

from sherlock.config import cfg


def balance(
    df: pd.DataFrame,
    per_party: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Rééquilibre le dataset en gardant au maximum `per_party`
    lignes par parti. Si un parti a moins de `per_party` lignes,
    on garde tout ce qu'il a (pas de sur-échantillonnage).

    Args:
        df:         DataFrame avec colonne 'parti'
        per_party:  max lignes par parti (défaut : cfg.balance.per_party)
        seed:       seed aléatoire (défaut : cfg.balance.seed)

    Returns:
        DataFrame équilibré et mélangé.
    """
    per_party = per_party or cfg.balance.per_party
    seed = seed or cfg.balance.seed

    if "parti" not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'parti'.")

    logger.info(f"Équilibrage : max {per_party} lignes par parti")

    before = df["parti"].value_counts()
    logger.debug(f"Distribution avant :\n{before.to_string()}")

    balanced = (
        df.groupby("parti", group_keys=False)
        .apply(lambda g: g.sample(
            n=min(len(g), per_party),
            random_state=seed,
        ))
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    after = balanced["parti"].value_counts()
    logger.info(f"Distribution après :\n{after.to_string()}")
    logger.info(f"Total : {len(balanced):,} lignes")

    return balanced