"""
Tests pour report.py (tableau de résultats généré depuis reports/).
"""

import json

import pandas as pd
import pytest

from sherlock.report import (
    END,
    START,
    render,
    render_compact,
    update_readme,
    update_results_doc,
)


@pytest.fixture
def reports_dir(tmp_path):
    run = tmp_path / "metrics" / "tfidf_logreg"
    run.mkdir(parents=True)
    result = {
        "overall": {"accuracy": 0.57, "f1_macro": 0.5695},
        "by_media": {"Twitter": {"f1_macro": 0.55}, "site_web": {"f1_macro": 0.65}},
    }
    (run / "test_metrics.json").write_text(json.dumps(result), encoding="utf-8")

    legacy = tmp_path / "legacy" / "camembert_ftV6"
    legacy.mkdir(parents=True)
    pd.DataFrame({"eval_f1_macro": [0.6289]}).to_csv(legacy / "overall_metrics.csv", index=False)
    return tmp_path


def test_render_uses_files_and_flags_missing_runs(reports_dir):
    table = render(reports_dir)
    assert "**0,570**" in table
    assert "*non lancé (`make train`)*" in table
    assert "**0,629**" in table


def test_render_compact_lists_only_existing_runs(reports_dir):
    table = render_compact(reports_dir)
    assert "**0,570**" in table
    assert "non lancé" not in table
    assert "nouveau fine-tuning" not in table


def test_update_results_doc_contains_full_tables(tmp_path, reports_dir):
    doc = tmp_path / "docs" / "results.md"
    update_results_doc(doc, reports_dir)
    text = doc.read_text(encoding="utf-8")
    assert text.startswith("# Résultats détaillés")
    assert "*non lancé (`make train`)*" in text
    assert "**0,629**" in text


def test_update_readme_replaces_only_block(tmp_path, reports_dir):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nancien\n{END}\nfin\n", encoding="utf-8")

    update_readme(readme, reports_dir)

    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n")
    assert text.endswith("fin\n")
    assert "ancien" not in text


def test_update_readme_requires_markers(tmp_path, reports_dir):
    readme = tmp_path / "README.md"
    readme.write_text("pas de marqueurs", encoding="utf-8")
    with pytest.raises(ValueError):
        update_readme(readme, reports_dir)
