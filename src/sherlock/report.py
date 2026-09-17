"""
report.py
---------
Génère le tableau de résultats du README à partir des fichiers de métriques.

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
    ("camembert_meta", "CamemBERT fine-tuné, texte + sentiment/ironie", "make train"),
    ("camembert_nometa", "CamemBERT fine-tuné, texte seul (ablation)", "make train-ablation"),
    ("legacy_v6_meta", "Modèle historique V6 ré-évalué avec le code actuel", "make eval-legacy"),
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
    """Remplace le bloc entre les marqueurs results:start / results:end."""
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError(f"Marqueurs {START} / {END} absents de {readme}")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    readme.write_text(f"{before}{START}\n{render(reports_dir)}\n{END}{after}", encoding="utf-8")
