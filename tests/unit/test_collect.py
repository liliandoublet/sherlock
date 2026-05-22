import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from sherlock.collect.base import BaseScraper
from sherlock.collect.wikipedia import WikipediaScraper


# ── Tests BaseScraper ─────────────────────────────────────────────────────────

def test_base_scraper_invalid_parti():
    """Lève une erreur si le parti n'est pas dans la liste."""
    with pytest.raises(ValueError, match="inconnu"):
        # On crée une sous-classe minimale pour tester BaseScraper
        class DummyScraper(BaseScraper):
            def fetch(self, **kwargs):
                return pd.DataFrame()
        DummyScraper(parti="PartiInexistant")

def test_base_scraper_valid_parti():
    """Accepte 'aucun' comme parti valide."""
    class DummyScraper(BaseScraper):
        def fetch(self, **kwargs):
            return pd.DataFrame()
    scraper = DummyScraper(parti="aucun")
    assert scraper.parti == "aucun"

def test_base_scraper_validate_missing_columns():
    """validate() lève une erreur si des colonnes manquent."""
    class DummyScraper(BaseScraper):
        def fetch(self, **kwargs):
            return pd.DataFrame()
    scraper = DummyScraper(parti="aucun")
    df_bad = pd.DataFrame({"texte": ["hello"], "parti": ["aucun"]})
    with pytest.raises(ValueError, match="colonnes manquantes"):
        scraper.validate(df_bad)

def test_base_scraper_validate_ok():
    """validate() retourne uniquement les colonnes du schéma."""
    class DummyScraper(BaseScraper):
        def fetch(self, **kwargs):
            return pd.DataFrame()
    scraper = DummyScraper(parti="aucun")
    df = pd.DataFrame([{
        "texte": "un texte",
        "compte": "titre",
        "parti": "aucun",
        "date": "2024-01-01",
        "source": "http://example.com",
        "media": "wikipedia",
        "colonne_extra": "à supprimer",
    }])
    result = scraper.validate(df)
    assert list(result.columns) == BaseScraper.SCHEMA
    assert "colonne_extra" not in result.columns


# ── Tests WikipediaScraper ────────────────────────────────────────────────────

def make_wiki_response(title: str, text: str) -> dict:
    """Crée une réponse API Wikipedia mockée."""
    return {
        "query": {
            "pages": {
                "1": {
                    "title": title,
                    "extract": text,
                    "fullurl": f"https://fr.wikipedia.org/wiki/{title}",
                    "revisions": [{"timestamp": "2024-01-01T00:00:00Z"}],
                }
            }
        }
    }

def test_wikipedia_fetch_article_valid():
    """_fetch_article retourne un dict valide pour un article correct."""
    scraper = WikipediaScraper()
    mock_response = make_wiki_response(
        "Tour Eiffel",
        "La tour Eiffel est une tour de fer puddlé " * 10,
    )
    with patch.object(scraper, "_api", return_value=mock_response):
        result = scraper._fetch_article("Tour Eiffel")
    assert result is not None
    assert result["parti"] == "aucun"
    assert result["media"] == "wikipedia"
    assert "Tour Eiffel" in result["compte"]

def test_wikipedia_fetch_article_too_short():
    """_fetch_article retourne None si le texte est trop court."""
    scraper = WikipediaScraper()
    mock_response = make_wiki_response("Stub", "Texte trop court.")
    with patch.object(scraper, "_api", return_value=mock_response):
        result = scraper._fetch_article("Stub")
    assert result is None

def test_wikipedia_fetch_article_missing():
    """_fetch_article retourne None si la page n'existe pas."""
    scraper = WikipediaScraper()
    mock_response = {
        "query": {"pages": {"-1": {"missing": "", "title": "Inexistant"}}}
    }
    with patch.object(scraper, "_api", return_value=mock_response):
        result = scraper._fetch_article("Inexistant")
    assert result is None