"""
site.py
-------
Données du site vitrine (site/index.html), une page statique publiée sur GitHub Pages.

Comme pour le README (report.py), aucun chiffre n'est saisi à la main : les métriques viennent de
reports/metrics/*/test_metrics.json et sont injectées dans la page, dans la balise
<script id="sherlock-data">. Avec un modèle, les exemples de la démo reçoivent en plus les
probabilités réellement calculées, dans les conditions de la démo (préfixe neutre, sans ironie).
"""

import json
import re
from pathlib import Path

from sherlock.config import cfg
from sherlock.demo import EXAMPLES, HESITATION, MAX_WORDS, MIN_WORDS

# Runs affichés par le site : conditions de la démo, avec annotations, baseline
RUNS = ("legacy_v6_public_input", "legacy_v6_meta", "tfidf_logreg")

DATA_TAG = re.compile(
    r'(<script id="sherlock-data" type="application/json">)(.*?)(</script>)', re.DOTALL
)


def summarize_run(result: dict) -> dict:
    """Garde ce que la page affiche : scores globaux, par média, par parti, matrice de confusion."""
    return {
        "n": result["n"],
        "accuracy": result["overall"]["accuracy"],
        "f1_macro": result["overall"]["f1_macro"],
        "by_media": {
            media: values["f1_macro"] for media, values in result.get("by_media", {}).items()
        },
        "labels": result["labels"],
        "per_party": {
            label: {
                "precision": result["report"][label]["precision"],
                "recall": result["report"][label]["recall"],
                "f1": result["report"][label]["f1-score"],
                "support": int(result["report"][label]["support"]),
            }
            for label in result["labels"]
        },
        "confusion_matrix": result["confusion_matrix"],
    }


def load_runs(metrics_dir: Path) -> dict[str, dict]:
    runs = {}
    for name in RUNS:
        path = metrics_dir / name / "test_metrics.json"
        if path.exists():
            runs[name] = summarize_run(json.loads(path.read_text(encoding="utf-8")))
    return runs


def model_label(source: str) -> str:
    """Nom affiché du modèle : le dossier pour un modèle local (pas de chemin personnel publié)."""
    path = Path(source)
    return path.name if path.exists() else source


def predict_examples(source: str) -> list[dict[str, float]]:
    """Probabilités par parti pour chaque exemple, comme la démo (sentiment neutre, sans ironie)."""
    from sherlock.model.predict import Predictor

    results = Predictor(source).predict([text for _, text in EXAMPLES])
    return [
        {party: round(score, 4) for party, score in result["all_scores"].items()}
        for result in results
    ]


def read_site_data(page: Path) -> dict:
    match = DATA_TAG.search(page.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f'Balise <script id="sherlock-data"> absente de {page}')
    payload = match.group(2).strip()
    return json.loads(payload) if payload else {}


def build_site_data(
    reports_dir: Path, model_source: str | None = None, previous: dict | None = None
) -> dict:
    """
    Assemble les données de la page. Sans modèle, les prédictions déjà présentes dans la page
    sont conservées pour les exemples inchangés : relancer sans modèle ne les efface pas.
    """
    if model_source:
        scores = predict_examples(model_source)
        examples_model = model_label(model_source)
    else:
        previous = previous or {}
        kept = {ex["text"]: ex.get("scores") for ex in previous.get("examples", [])}
        scores = [kept.get(text) for _, text in EXAMPLES]
        examples_model = previous.get("examples_model") if any(scores) else None

    return {
        "version": cfg.project.version,
        "thresholds": {"hesitation": HESITATION, "min_words": MIN_WORDS, "max_words": MAX_WORDS},
        "runs": load_runs(reports_dir / "metrics"),
        "examples": [
            {"label": label, "text": text, "scores": example_scores}
            for (label, text), example_scores in zip(EXAMPLES, scores, strict=True)
        ],
        "examples_model": examples_model,
    }


def update_site(page: Path, reports_dir: Path, model_source: str | None = None) -> dict:
    """Réécrit uniquement le contenu de la balise de données de la page."""
    data = build_site_data(reports_dir, model_source, previous=read_site_data(page))
    # "</" échappé : le JSON ne peut pas fermer la balise <script> qui le contient
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = DATA_TAG.sub(lambda m: f"{m.group(1)}{payload}{m.group(3)}", page.read_text("utf-8"))
    page.write_text(html, encoding="utf-8")
    return data
