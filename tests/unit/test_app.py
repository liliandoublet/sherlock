"""
Tests de l'appli Streamlit publique (AppTest), sur un mini-modèle local : aucun téléchargement.
"""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from tests.unit.tiny_model import PARTIES, save_tiny_model

APP = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Lance l'appli sur un mini-modèle ; le modèle est choisi via SHERLOCK_MODEL."""
    monkeypatch.setenv("SHERLOCK_MODEL", str(save_tiny_model(tmp_path / "model")))
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    return at


def click(at, label: str):
    button = next(b for b in at.button if b.label == label)
    button.click().run()
    return at


def result_metrics(at):
    """La métrique du parti prédit (l'onglet Performances a aussi des métriques)."""
    return [m for m in at.metric if m.label in ("Parti le plus probable", "Piste la plus probable")]


def test_app_starts_without_error(app):
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Essayer", "Performances", "À propos"]
    assert app.title[0].value.endswith("Sherlock")


def test_example_gives_a_prediction(app):
    click(app, "Retraite à 60 ans")
    assert not app.exception
    assert result_metrics(app)[0].value in PARTIES
    assert "retraite" in app.text_area[0].value.lower()


def test_low_confidence_is_flagged(app):
    """Un modèle non entraîné répond ~1/8 partout : l'appli doit dire qu'il hésite."""
    click(app, "Sans signal politique")
    assert any("hésite" in warning.value for warning in app.warning)
    assert result_metrics(app)[0].label == "Piste la plus probable"


def test_empty_text_asks_for_input(app):
    click(app, "Analyser")
    assert any("Écris ou colle" in warning.value for warning in app.warning)
    assert not result_metrics(app)


def test_missing_model_shows_help_not_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("SHERLOCK_MODEL", str(tmp_path / "absent"))
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    click(at, "Retraite à 60 ans")
    assert not at.exception
    assert any("Modèle introuvable" in error.value for error in at.error)
    assert any("make demo-local" in info.value for info in at.info)


def test_performance_tab_reads_versioned_metrics(app):
    """Les métriques de reports/ sont versionnées : l'onglet doit afficher ses graphiques."""
    assert len(app.get("plotly_chart")) >= 2
