from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedShuffleSplit

from sherlock.config import cfg


def split(
    df: pd.DataFrame,
    train_size: float | None = None,
    val_size: float | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split stratifié sur 'parti'.

    Returns:
        (train_df, val_df, test_df)
    """
    train_size = train_size or cfg.split.train
    val_size = val_size or cfg.split.val
    seed = seed or cfg.split.seed

    if "parti" not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'parti'.")

    logger.info(f"Split {train_size:.0%} / {val_size:.0%} / {1 - train_size - val_size:.0%}")

    # Split 1 : train vs (val + test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - train_size,
        random_state=seed,
    )
    train_idx, temp_idx = next(sss1.split(df, df["parti"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Split 2 : val vs test
    val_fraction = val_size / (1 - train_size)
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - val_fraction,
        random_state=seed,
    )
    val_idx, test_idx = next(sss2.split(temp_df, temp_df["parti"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    logger.info(f"Résultat : {len(train_df):,} train / {len(val_df):,} val / {len(test_df):,} test")
    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Sauvegarde les splits en parquet (plus rapide que CSV)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)
    logger.info(f"Splits sauvegardés dans {out_dir}")
