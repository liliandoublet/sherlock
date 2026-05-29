from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sherlock.annotate.irony import annotate_irony
from sherlock.annotate.pipeline import annotate
from sherlock.annotate.sentiment import annotate_sentiment

# ── Fixture commune ───────────────────────────────────────────────────────────


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "texte": [
                "Le gouvernement annonce de nouvelles mesures économiques.",
                "Quelle brillante idée de supprimer les services publics.",
                "La réforme des retraites divise profondément la société.",
            ],
            "parti": ["Renaissance", "LFI", "PS"],
            "media": ["twitter", "twitter", "web"],
        }
    )


def make_mock_pipeline(label: str, score: float = 0.95):
    """Crée un mock de pipeline HuggingFace."""
    mock = MagicMock()

    mock.side_effect = lambda texts, **kwargs: [{"label": label, "score": score} for _ in texts]
    return mock


# ── Tests sentiment ───────────────────────────────────────────────────────────


def test_sentiment_adds_columns(sample_df):
    """annotate_sentiment ajoute les colonnes sentiment et sentiment_score."""
    mock_pipeline = make_mock_pipeline("POSITIVE")
    with patch("sherlock.annotate.sentiment.pipeline", return_value=mock_pipeline):
        result = annotate_sentiment(sample_df, batch_size=8)
    assert "sentiment" in result.columns
    assert "sentiment_score" in result.columns


def test_sentiment_valid_labels(sample_df):
    """Les labels sentiment sont POSITIVE ou NEGATIVE."""
    mock_pipeline = make_mock_pipeline("POSITIVE")
    with patch("sherlock.annotate.sentiment.pipeline", return_value=mock_pipeline):
        result = annotate_sentiment(sample_df, batch_size=8)
    assert set(result["sentiment"].unique()).issubset({"POSITIVE", "NEGATIVE"})


def test_sentiment_score_range(sample_df):
    """Les scores sont entre 0 et 1."""
    mock_pipeline = make_mock_pipeline("POSITIVE", score=0.87)
    with patch("sherlock.annotate.sentiment.pipeline", return_value=mock_pipeline):
        result = annotate_sentiment(sample_df, batch_size=8)
    assert result["sentiment_score"].between(0, 1).all()


def test_sentiment_missing_column(sample_df):
    """Lève une erreur si la colonne texte est absente."""
    with pytest.raises(ValueError, match="absente"):
        annotate_sentiment(sample_df, text_col="colonne_inexistante")


def test_sentiment_preserves_rows(sample_df):
    """Le nombre de lignes ne change pas."""
    mock_pipeline = make_mock_pipeline("NEGATIVE")
    with patch("sherlock.annotate.sentiment.pipeline", return_value=mock_pipeline):
        result = annotate_sentiment(sample_df, batch_size=8)
    assert len(result) == len(sample_df)


# ── Tests ironie ──────────────────────────────────────────────────────────────


def test_irony_adds_columns(sample_df):
    """annotate_irony ajoute les colonnes ironie et ironie_score."""
    mock_pipeline = make_mock_pipeline("not_ironic")
    with patch("sherlock.annotate.irony.pipeline", return_value=mock_pipeline):
        result = annotate_irony(sample_df, batch_size=8)
    assert "ironie" in result.columns
    assert "ironie_score" in result.columns


def test_irony_missing_column(sample_df):
    """Lève une erreur si la colonne texte est absente."""
    with pytest.raises(ValueError, match="absente"):
        annotate_irony(sample_df, text_col="colonne_inexistante")


def test_irony_preserves_rows(sample_df):
    """Le nombre de lignes ne change pas."""
    mock_pipeline = make_mock_pipeline("ironic")
    with patch("sherlock.annotate.irony.pipeline", return_value=mock_pipeline):
        result = annotate_irony(sample_df, batch_size=8)
    assert len(result) == len(sample_df)


# ── Tests pipeline complet ────────────────────────────────────────────────────


def test_annotate_pipeline_full(sample_df):
    """Le pipeline complet ajoute les 4 colonnes d'annotation."""
    mock_sent = make_mock_pipeline("POSITIVE")
    mock_irony = make_mock_pipeline("not_ironic")
    with (
        patch("sherlock.annotate.sentiment.pipeline", return_value=mock_sent),
        patch("sherlock.annotate.irony.pipeline", return_value=mock_irony),
    ):
        result = annotate(sample_df, batch_size=8)
    expected_cols = {"sentiment", "sentiment_score", "ironie", "ironie_score"}
    assert expected_cols.issubset(set(result.columns))
    assert len(result) == len(sample_df)
