"""
Tests pour dataset/legacy.py (import des splits historiques).
"""

import pandas as pd
import pytest

from sherlock.dataset.legacy import import_legacy, normalize, overlap_report


def make_split(texts: list[str], parti: str = "RN") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "texte": texts,
            "compte": "x",
            "parti": parti,
            "date": "",
            "source": "",
            "media": "Twitter",
            "ironie": ["False"] * len(texts),
            "sentiment": ["positre"] * len(texts),
            "parti_media": f"{parti}_Twitter",
        }
    )


def test_normalize_fixes_sentiment_typo_and_bool():
    df = normalize(make_split(["un texte"]))
    assert df.loc[0, "sentiment"] == "positif"
    # astype(bool) sur la chaîne "False" donnerait True
    assert not df.loc[0, "ironie"]


def test_normalize_rejects_unknown_party():
    with pytest.raises(ValueError, match="Partis inconnus"):
        normalize(make_split(["texte"], parti="MoDem"))


def test_normalize_rejects_missing_columns():
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        normalize(pd.DataFrame({"texte": ["a"]}))


def test_overlap_report_counts_shared_texts():
    splits = {
        "train": normalize(make_split(["a", "b", "c"])),
        "val": normalize(make_split(["c", "d"])),
        "test": normalize(make_split(["a", "e"])),
    }
    assert overlap_report(splits) == {"train_val": 1, "train_test": 1, "val_test": 0}


def test_import_legacy_writes_parquet(tmp_path):
    src = tmp_path / "legacy"
    src.mkdir()
    for name, texts in [("train", ["a", "b"]), ("val", ["c"]), ("test", ["d"])]:
        make_split(texts).to_csv(src / f"{name}.csv", sep="|", index=False)

    stats = import_legacy(src, tmp_path / "processed")

    assert stats["train"] == 2
    assert stats["overlap_train_test"] == 0
    df = pd.read_parquet(tmp_path / "processed" / "train.parquet")
    assert df["ironie"].dtype == bool
