"""
Tests pour site.py (données du site vitrine injectées dans site/index.html).
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sherlock.cli import app
from sherlock.demo import EXAMPLES
from sherlock.site import build_site_data, load_runs, read_site_data, update_site

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["EELV", "LFI"]

PAGE = """<!doctype html><title>Sherlock</title>
<script id="sherlock-data" type="application/json"></script>
<script>console.log("intact")</script>
"""


def fake_metrics(f1: float) -> dict:
    return {
        "n": 4,
        "overall": {"accuracy": 0.75, "f1_macro": f1},
        "by_media": {"Twitter": {"f1_macro": 0.7}, "site_web": {"f1_macro": 0.8}},
        "labels": LABELS,
        "report": {
            label: {"precision": 0.8, "recall": 0.7, "f1-score": 0.75, "support": 2.0}
            for label in LABELS
        },
        "confusion_matrix": [[2, 0], [1, 1]],
    }


@pytest.fixture
def reports_dir(tmp_path):
    for name, f1 in [("legacy_v6_public_input", 0.626), ("tfidf_logreg", 0.57)]:
        run = tmp_path / "reports" / "metrics" / name
        run.mkdir(parents=True)
        (run / "test_metrics.json").write_text(json.dumps(fake_metrics(f1)), encoding="utf-8")
    return tmp_path / "reports"


@pytest.fixture
def page(tmp_path):
    path = tmp_path / "index.html"
    path.write_text(PAGE, encoding="utf-8")
    return path


def test_build_site_data_reads_metrics_files(reports_dir):
    data = build_site_data(reports_dir)
    assert set(data["runs"]) == {"legacy_v6_public_input", "tfidf_logreg"}
    run = data["runs"]["legacy_v6_public_input"]
    assert run["f1_macro"] == 0.626
    assert run["by_media"] == {"Twitter": 0.7, "site_web": 0.8}
    assert run["per_party"]["LFI"] == {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 2}
    assert run["confusion_matrix"] == [[2, 0], [1, 1]]


def test_examples_are_the_demo_examples_without_scores(reports_dir):
    data = build_site_data(reports_dir)
    assert [(ex["label"], ex["text"]) for ex in data["examples"]] == EXAMPLES
    assert all(ex["scores"] is None for ex in data["examples"])
    assert data["examples_model"] is None


def test_update_site_only_rewrites_data_tag(page, reports_dir):
    update_site(page, reports_dir)
    html = page.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html><title>Sherlock</title>")
    assert 'console.log("intact")' in html
    assert read_site_data(page)["runs"]["tfidf_logreg"]["f1_macro"] == 0.57


def test_payload_cannot_close_the_script_tag(page, reports_dir, monkeypatch):
    monkeypatch.setattr("sherlock.site.EXAMPLES", [("Piège", "fin du texte </script><b>")])
    update_site(page, reports_dir)
    html = page.read_text(encoding="utf-8")
    assert "</script><b>" not in html
    assert read_site_data(page)["examples"][0]["text"] == "fin du texte </script><b>"


def test_rerun_without_model_keeps_previous_predictions(page, reports_dir):
    scores = {"EELV": 0.9, "LFI": 0.1}
    previous = {
        "examples": [{"label": "x", "text": EXAMPLES[0][1], "scores": scores}],
        "examples_model": "camembert_v6_meta",
    }
    data = build_site_data(reports_dir, previous=previous)
    assert data["examples"][0]["scores"] == scores
    assert data["examples"][1]["scores"] is None
    assert data["examples_model"] == "camembert_v6_meta"


def test_missing_data_tag_raises(tmp_path, reports_dir):
    bare = tmp_path / "bare.html"
    bare.write_text("<p>pas de données</p>", encoding="utf-8")
    with pytest.raises(ValueError, match="sherlock-data"):
        update_site(bare, reports_dir)


def test_cli_site_updates_page(page, reports_dir):
    result = CliRunner().invoke(
        app, ["site", "--page", str(page), "--reports-dir", str(reports_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "0/6 exemples" in " ".join(result.output.split())  # rich coupe les lignes longues
    assert read_site_data(page)["runs"]


def test_published_site_matches_versioned_metrics():
    """site/index.html doit refléter reports/metrics : sinon, lancer `make site`."""
    published = read_site_data(ROOT / "site" / "index.html")
    assert published["runs"] == load_runs(ROOT / "reports" / "metrics")


def test_examples_get_real_probabilities_from_a_model(tmp_path, page, reports_dir):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.unit.tiny_model import PARTIES, save_tiny_model

    model_dir = save_tiny_model(tmp_path / "model")
    data = update_site(page, reports_dir, model_source=str(model_dir))
    assert data["examples_model"] == "model"  # nom du dossier, pas le chemin complet
    for example in data["examples"]:
        assert set(example["scores"]) == set(PARTIES)
        assert sum(example["scores"].values()) == pytest.approx(1, abs=1e-3)
