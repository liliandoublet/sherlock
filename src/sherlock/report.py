"""
report.py
---------
Génère les tableaux de résultats (README compact, docs/results.md complet) à partir des
fichiers de métriques.

Aucun chiffre n'est saisi à la main : tout vient de reports/metrics/*/test_metrics.json
(runs MLflow) et de reports/legacy/ (CSV exportés par l'ancien projet).
"""

import json
from pathlib import Path

import pandas as pd

START = "<!-- results:start -->"
END = "<!-- results:end -->"

# (dossier de run, libellé, commande qui le produit)
KNOWN_RUNS = [
    (
        "legacy_v6_public_input",
        "V6 dans les conditions de la démo (sans annotations)",
        "make eval-legacy",
    ),
    ("camembert_meta", "CamemBERT fine-tuné, texte + sentiment/ironie", "make train"),
    ("camembert_nometa", "CamemBERT fine-tuné, texte seul (ablation)", "make train-ablation"),
    ("legacy_v6_meta", "V6 avec sentiment/ironie annotés (référence du CV)", "make eval-legacy"),
    ("tfidf_logreg", "Baseline TF-IDF + régression logistique", "make baselines"),
]

LEGACY_NOTES = {
    "camembert_ftV4": "dossier nommé « sans sentiment/ironie » mais évalué avec (cf. décision 008)",
    "camembert_ftV6": "CamemBERT + préfixes sentiment/ironie, modèle ré-évalué ci-dessus",
    "camembert_ftV8": "CamemBERT + sentiment/ironie en features numériques",
    "camembert_ftV9": "EuroBERT (pas CamemBERT)",
}


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}".replace(".", ",")


def current_rows(metrics_dir: Path) -> list[str]:
    rows = []
    known = {name for name, _, _ in KNOWN_RUNS}
    extra = sorted(p.name for p in metrics_dir.glob("*") if p.is_dir() and p.name not in known)
    for name, label, command in [*KNOWN_RUNS, *[(n, n, "—") for n in extra]]:
        path = metrics_dir / name / "test_metrics.json"
        if not path.exists():
            rows.append(f"| {label} | `{name}` | *non lancé (`{command}`)* | | | |")
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        overall, media = result["overall"], result.get("by_media", {})
        rows.append(
            f"| {label} | `{name}` | {_fmt(overall['accuracy'])} | **{_fmt(overall['f1_macro'])}** "
            f"| {_fmt(media.get('Twitter', {}).get('f1_macro'))} "
            f"| {_fmt(media.get('site_web', {}).get('f1_macro'))} |"
        )
    return rows


def legacy_rows(legacy_dir: Path) -> list[str]:
    rows = []
    for version_dir in sorted(legacy_dir.glob("camembert_ftV*")):
        overall_csv = next(version_dir.glob("*overall*.csv"), None)
        if overall_csv is None:
            continue
        f1 = pd.read_csv(overall_csv)["eval_f1_macro"].iloc[0]
        media_csv = version_dir / "metrics_by_media.csv"
        by_media = (
            pd.read_csv(media_csv).set_index("media")["f1_macro"].to_dict()
            if media_csv.exists()
            else {}
        )
        rows.append(
            f"| {version_dir.name.removeprefix('camembert_ft')} | **{_fmt(f1)}** "
            f"| {_fmt(by_media.get('Twitter'))} | {_fmt(by_media.get('site_web'))} "
            f"| {LEGACY_NOTES.get(version_dir.name, '')} |"
        )
    return rows


COMPACT_LABELS = {
    "legacy_v6_public_input": "Sherlock (V6), conditions de la démo",
    "legacy_v6_meta": "Sherlock (V6) avec sentiment/ironie annotés",
    "camembert_meta": "Sherlock, nouveau fine-tuning",
    "camembert_nometa": "Sherlock, nouveau fine-tuning sans annotations",
    "tfidf_logreg": "Baseline TF-IDF + régression logistique",
}


def render_compact(reports_dir: Path) -> str:
    """Tableau court pour le README : uniquement les runs qui existent."""
    metrics_dir = reports_dir / "metrics"
    lines = [
        "| Modèle | F1 macro | Accuracy | F1 tweets | F1 sites officiels |",
        "|---|---|---|---|---|",
    ]
    for name, label in COMPACT_LABELS.items():
        path = metrics_dir / name / "test_metrics.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        overall, media = result["overall"], result.get("by_media", {})
        lines.append(
            f"| {label} | **{_fmt(overall['f1_macro'])}** | {_fmt(overall['accuracy'])} "
            f"| {_fmt(media.get('Twitter', {}).get('f1_macro'))} "
            f"| {_fmt(media.get('site_web', {}).get('f1_macro'))} |"
        )
    return "\n".join(lines)


def render(reports_dir: Path) -> str:
    lines = [
        "**Runs du pipeline actuel** (test set, 1 470 textes, suivis dans MLflow) :",
        "",
        "| Modèle | Run | Accuracy | F1 macro | F1 Twitter | F1 site web |",
        "|---|---|---|---|---|---|",
        *current_rows(reports_dir / "metrics"),
        "",
        "**Historique du projet de fin d'études** (même test set, transformers 4.52, "
        "exports CSV dans `reports/legacy/`) :",
        "",
        "| Version | F1 macro | F1 Twitter | F1 site web | Note |",
        "|---|---|---|---|---|",
        *legacy_rows(reports_dir / "legacy"),
    ]
    return "\n".join(lines)


def update_readme(readme: Path, reports_dir: Path) -> None:
    """Remplace le bloc entre les marqueurs results:start / results:end par le tableau compact."""
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError(f"Marqueurs {START} / {END} absents de {readme}")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    block = render_compact(reports_dir)
    readme.write_text(f"{before}{START}\n{block}\n{END}{after}", encoding="utf-8")


RESULTS_HEADER = """# Résultats détaillés

Page **générée** par `make report` à partir de `reports/` : ne pas modifier à la main.
"""


def update_results_doc(path: Path, reports_dir: Path) -> None:
    """Écrit la page de résultats complète (runs actuels + historique du projet de fin d'études)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{RESULTS_HEADER}\n{render(reports_dir)}\n", encoding="utf-8")
