"""
Tests pour space.py (copie du dépôt envoyée au Space Hugging Face de la démo en ligne).
"""

from pathlib import Path

import pytest

from sherlock.space import stage_space

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def staged(tmp_path):
    return stage_space(tmp_path / "space", model="quelquun/sherlock-test")


def test_space_gets_the_docker_files_at_its_root(staged):
    assert (staged / "Dockerfile").read_text() == (ROOT / "space" / "Dockerfile").read_text()
    card = (staged / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\n")
    assert "sdk: docker" in card
    assert "app_port: 8501" in card
    assert "  - quelquun/sherlock-test" in card
    assert "__MODEL__" not in card


def test_space_contains_what_the_image_needs(staged):
    for path in [
        "pyproject.toml",
        "params.yaml",
        "src/sherlock/demo.py",
        "app/streamlit_app.py",
        "reports/metrics/legacy_v6_public_input/test_metrics.json",
        ".streamlit/config.toml",
    ]:
        assert (staged / path).exists(), path
    assert not list(staged.rglob("__pycache__"))
    assert not (staged / "tests").exists()


def test_staging_twice_starts_from_scratch(tmp_path):
    build = tmp_path / "space"
    stage_space(build, model="a/b")
    (build / "obsolete.txt").write_text("x")
    stage_space(build, model="a/b")
    assert not (build / "obsolete.txt").exists()


def test_space_dockerfile_runs_the_app_with_the_downloaded_model():
    dockerfile = (ROOT / "space" / "Dockerfile").read_text(encoding="utf-8")
    assert "snapshot_download(cfg.demo.model" in dockerfile
    assert "SHERLOCK_MODEL=/app/model" in dockerfile
    assert "SHERLOCK_HOSTED=1" in dockerfile
    assert "--server.port=8501" in dockerfile
