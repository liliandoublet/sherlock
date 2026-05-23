"""
cli.py
------
Point d'entrée principal de sherlock.
Usage :
    sherlock --help
    sherlock clean --source twitter --input data/raw/twitter/RN.csv
    sherlock annotate --input data/interim/merged.csv
    sherlock dataset merge --inputs data/interim/twitter.csv data/interim/web.csv
    sherlock dataset split --input data/interim/merged.csv
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

from sherlock.logging import setup_logging, get_logger

# ── App principale ────────────────────────────────────────────────────────────
app = typer.Typer(
    name="sherlock",
    help="French political ideology classifier.",
    add_completion=False,
    rich_markup_mode="rich",
)

# ── Sous-apps ─────────────────────────────────────────────────────────────────
dataset_app = typer.Typer(help="Construire et préparer le dataset.")
app.add_typer(dataset_app, name="dataset")

console = Console()
logger  = get_logger(__name__)


# ── Callback global ───────────────────────────────────────────────────────────

@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Mode verbeux."),
    log_dir: str  = typer.Option("logs", "--log-dir", help="Dossier des logs."),
):
    """Sherlock - détection d'idéologie politique dans les textes français."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(log_dir=log_dir, level=level)


# ── Commande : info ───────────────────────────────────────────────────────────

@app.command()
def info():
    """Affiche la configuration courante du projet."""
    from sherlock.config import cfg

    table = Table(title="Sherlock - Configuration", show_header=True)
    table.add_column("Paramètre", style="cyan")
    table.add_column("Valeur", style="green")

    table.add_row("Version",    cfg.project.version)
    table.add_row("Langue",     cfg.project.language)
    table.add_row("Partis",     ", ".join(cfg.parties))
    table.add_row("Séparateur", repr(cfg.io.separator))
    table.add_row("Modèle ML",  cfg.model.name)
    table.add_row("Chunk size", str(cfg.cleaning.chunk_size))
    table.add_row("Per party",  str(cfg.balance.per_party))

    console.print(table)


# ── Commande : clean ──────────────────────────────────────────────────────────

@app.command()
def clean(
    input:  Path = typer.Argument(..., help="CSV source à nettoyer."),
    output: Path = typer.Argument(..., help="CSV de sortie nettoyé."),
    source: str  = typer.Option("twitter", help="'twitter' ou 'web'."),
):
    """Nettoie un CSV de tweets ou de communiqués web."""
    import pandas as pd
    from sherlock.clean.pipeline import run as clean_pipeline
    from sherlock.config import cfg

    if not input.exists():
        console.print(f"[red]Fichier introuvable : {input}[/red]")
        raise typer.Exit(1)

    df = pd.read_csv(input, sep=cfg.io.separator, encoding=cfg.io.encoding, dtype=str)
    console.print(f"[cyan]Nettoyage de {len(df):,} lignes ({source})...[/cyan]")

    df_clean = clean_pipeline(df, source=source)

    output.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output, sep=cfg.io.separator, encoding=cfg.io.encoding, index=False)
    console.print(f"[green]Sauvegardé : {output} ({len(df_clean):,} lignes)[/green]")


# ── Commande : annotate ───────────────────────────────────────────────────────

@app.command()
def annotate(
    input:      Path = typer.Argument(..., help="CSV à annoter."),
    output:     Path = typer.Argument(..., help="CSV annoté en sortie."),
    batch_size: int  = typer.Option(32, help="Taille des batchs."),
):
    """Annote sentiment et ironie sur un CSV nettoyé."""
    import pandas as pd
    from sherlock.annotate.pipeline import annotate as run_annotate, save_annotated
    from sherlock.config import cfg

    if not input.exists():
        console.print(f"[red]Fichier introuvable : {input}[/red]")
        raise typer.Exit(1)

    df = pd.read_csv(input, sep=cfg.io.separator, encoding=cfg.io.encoding)
    console.print(f"[cyan]Annotation de {len(df):,} lignes...[/cyan]")

    df_annotated = run_annotate(df, batch_size=batch_size)
    save_annotated(df_annotated, output)
    console.print(f"[green]Annoté : {output}[/green]")


# ── Sous-commandes dataset ────────────────────────────────────────────────────

@dataset_app.command("merge")
def dataset_merge(
    inputs:  list[Path] = typer.Argument(..., help="CSV sources à fusionner."),
    output:  Path       = typer.Option(..., "--output", "-o", help="CSV de sortie."),
):
    """Fusionne plusieurs CSV sources en un seul dataset."""
    from sherlock.dataset.merge import merge_sources

    console.print(f"[cyan]Fusion de {len(inputs)} fichiers...[/cyan]")
    df = merge_sources(*inputs)
    output.parent.mkdir(parents=True, exist_ok=True)

    from sherlock.config import cfg
    df.to_csv(output, sep=cfg.io.separator, encoding=cfg.io.encoding, index=False)
    console.print(f"[green]Fusionné : {output} ({len(df):,} lignes)[/green]")


@dataset_app.command("balance")
def dataset_balance(
    input:      Path = typer.Argument(..., help="CSV à équilibrer."),
    output:     Path = typer.Argument(..., help="CSV équilibré en sortie."),
    per_party:  int  = typer.Option(1392, help="Lignes max par parti."),
):
    """Équilibre le dataset par parti."""
    import pandas as pd
    from sherlock.dataset.balance import balance
    from sherlock.config import cfg

    df = pd.read_csv(input, sep=cfg.io.separator, encoding=cfg.io.encoding)
    console.print(f"[cyan]Équilibrage à {per_party} lignes/parti...[/cyan]")
    df_balanced = balance(df, per_party=per_party)
    df_balanced.to_csv(output, sep=cfg.io.separator, encoding=cfg.io.encoding, index=False)
    console.print(f"[green]Équilibré : {output} ({len(df_balanced):,} lignes)[/green]")

# ── Commande : train ──────────────────────────────────────────────────────────

@app.command()
def train(
    data_dir:  Path = typer.Option(
        Path("data/processed"), "--data-dir", help="Dossier des splits parquet."
    ),
    output_dir: Path = typer.Option(
        Path("models/camembert_party"), "--output-dir", help="Dossier de sortie du modèle."
    ),
    run_name: str = typer.Option("camembert_party", "--run-name", help="Nom du run MLflow."),
):
    """Fine-tune CamemBERTa-v2 pour la classification de parti."""
    from sherlock.model.train import train as run_train
    console.print("[cyan]Démarrage de l'entraînement...[/cyan]")
    metrics = run_train(data_dir=data_dir, output_dir=output_dir, run_name=run_name)
    console.print(f"[green]Test F1 macro : {metrics['f1_macro']}[/green]")
    console.print("[dim]Visualiser les runs : mlflow ui[/dim]")


# ── Commande : predict ────────────────────────────────────────────────────────

@app.command()
def predict(
    text: str = typer.Argument(..., help="Texte à classifier."),
    model_path: Path = typer.Option(
        Path("models/camembert_party/best_model.pt"),
        "--model-path",
        help="Chemin vers le modèle sauvegardé.",
    ),
):
    """Prédit le parti politique d'un texte."""
    from sherlock.model.predict import predict_text
    result = predict_text(text=text, model_path=model_path)

    table = Table(title="Prédiction", show_header=True)
    table.add_column("Parti", style="cyan")
    table.add_column("Score", style="green")

    for parti, score in sorted(
        result["all_scores"].items(), key=lambda x: x[1], reverse=True
    ):
        style = "bold green" if parti == result["parti"] else ""
        table.add_row(parti, f"{score:.1%}", style=style)

    console.print(table)
    
@dataset_app.command("split")
def dataset_split(
    input:  Path = typer.Argument(..., help="CSV à splitter."),
    outdir: Path = typer.Option(
        "data/processed", "--outdir", "-d", help="Dossier de sortie."
    ),
):
    """Crée les splits train/val/test stratifiés."""
    import pandas as pd
    from sherlock.dataset.split import split, save_splits
    from sherlock.config import cfg

    df = pd.read_csv(input, sep=cfg.io.separator, encoding=cfg.io.encoding)
    console.print(f"[cyan]Split stratifié de {len(df):,} lignes...[/cyan]")
    train, val, test = split(df)
    save_splits(train, val, test, outdir)
    console.print(
        f"[green]Splits : {len(train):,} train / "
        f"{len(val):,} val / {len(test):,} test[/green]"
    )