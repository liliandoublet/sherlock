import pandas as pd
import pytest

from sherlock.dataset.balance import balance
from sherlock.dataset.split import split

# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_df(n_per_party: dict[str, int]) -> pd.DataFrame:
    """Crée un DataFrame de test avec n lignes par parti."""
    rows = []
    for parti, n in n_per_party.items():
        for i in range(n):
            rows.append(
                {
                    "texte": f"texte {parti} numéro {i} avec assez de mots",
                    "parti": parti,
                    "compte": f"@{parti.lower()}",
                    "date": "2024-01-01",
                    "source": "test",
                    "media": "twitter",
                }
            )
    return pd.DataFrame(rows)


# ── Tests balance ─────────────────────────────────────────────────────────────


def test_balance_equal_distribution():
    """Après équilibrage, chaque parti a le même nombre de lignes."""
    df = make_df({"LFI": 100, "RN": 80, "PS": 60})
    result = balance(df, per_party=50, seed=42)
    counts = result["parti"].value_counts()
    assert counts.nunique() == 1, f"Distribution inégale : {counts.to_dict()}"


def test_balance_respects_per_party():
    """Aucun parti ne dépasse per_party lignes."""
    df = make_df({"LFI": 200, "RN": 200})
    result = balance(df, per_party=100, seed=42)
    assert all(result["parti"].value_counts() <= 100)


def test_balance_keeps_small_party():
    """Un parti avec moins de per_party lignes garde tout."""
    df = make_df({"LFI": 200, "PCF": 30})
    result = balance(df, per_party=100, seed=42)
    assert result[result["parti"] == "PCF"].shape[0] == 30


def test_balance_missing_column():
    """Lève une erreur si la colonne 'parti' est absente."""
    df = pd.DataFrame({"texte": ["a", "b"]})
    with pytest.raises(ValueError, match="parti"):
        balance(df)


# ── Tests split ───────────────────────────────────────────────────────────────


def test_split_sizes():
    """Les tailles des splits respectent les ratios."""
    df = make_df({"LFI": 100, "RN": 100, "PS": 100})
    train, val, test = split(df, train_size=0.8, val_size=0.1, seed=42)
    total = len(train) + len(val) + len(test)
    assert total == len(df)
    assert abs(len(train) / total - 0.8) < 0.05


def test_split_no_overlap():
    """Aucun texte n'apparaît dans deux splits."""
    df = make_df({"LFI": 100, "RN": 100, "PS": 100})
    train, val, test = split(df, train_size=0.8, val_size=0.1, seed=42)
    train_texts = set(train["texte"])
    val_texts = set(val["texte"])
    test_texts = set(test["texte"])
    assert train_texts.isdisjoint(val_texts)
    assert train_texts.isdisjoint(test_texts)
    assert val_texts.isdisjoint(test_texts)


def test_split_stratified():
    """Chaque parti est représenté dans train, val et test."""
    df = make_df({"LFI": 100, "RN": 100, "PS": 100})
    train, val, test = split(df, train_size=0.8, val_size=0.1, seed=42)
    for split_df in (train, val, test):
        assert set(split_df["parti"]) == {"LFI", "RN", "PS"}


def test_split_missing_column():
    """Lève une erreur si la colonne 'parti' est absente."""
    df = pd.DataFrame({"texte": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="parti"):
        split(df)
