"""
evaluate.py
-----------
Évaluation d'un modèle sur un split : métriques globales, par média,
rapport de classification et matrice de confusion.

Sert à la fois en fin d'entraînement et pour ré-évaluer un modèle existant
(par exemple le modèle historique V6) avec exactement le même code.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sherlock.config import cfg
from sherlock.model.features import build_inputs


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_proba(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 32,
    max_length: int | None = None,
) -> np.ndarray:
    """Probabilités (n_textes, n_classes). Les textes sont triés par longueur pour limiter le padding."""
    device = next(model.parameters()).device
    max_length = max_length or cfg.model.max_length
    model.eval()

    order = np.argsort([len(t) for t in texts])
    probs = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)

    for start in range(0, len(texts), batch_size):
        idx = order[start : start + batch_size]
        enc = tokenizer(
            [texts[i] for i in idx],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**enc).logits
        probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()

    return probs


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def evaluate_dataframe(
    model,
    tokenizer,
    df: pd.DataFrame,
    use_meta: bool,
    batch_size: int = 32,
) -> dict:
    """Évalue le modèle sur un DataFrame (colonnes texte, parti, media, sentiment, ironie)."""
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    label2id = {label: i for i, label in enumerate(labels)}

    probs = predict_proba(model, tokenizer, build_inputs(df, use_meta), batch_size=batch_size)
    y_true = df["parti"].map(label2id).to_numpy()
    y_pred = probs.argmax(axis=1)

    result = {
        "n": len(df),
        "overall": compute_metrics(y_true, y_pred),
        "by_media": {},
        "report": classification_report(
            y_true, y_pred, labels=range(len(labels)), target_names=labels, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(labels))).tolist(),
        "labels": labels,
    }
    if "media" in df.columns:
        for media, mask in df.groupby("media").groups.items():
            pos = df.index.get_indexer(mask)
            result["by_media"][media] = compute_metrics(y_true[pos], y_pred[pos])

    overall = result["overall"]
    logger.info(f"accuracy={overall['accuracy']:.4f} | f1_macro={overall['f1_macro']:.4f}")
    return result


def plot_confusion_matrix(result: dict, path: Path) -> Path:
    """Sauvegarde la matrice de confusion normalisée en PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    cm = np.array(result["confusion_matrix"], dtype=float)
    cm_pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(cm_pct, display_labels=result["labels"]).plot(
        cmap="Blues", xticks_rotation=45, values_format=".2f", colorbar=False, ax=ax
    )
    ax.set_title(f"F1 macro = {result['overall']['f1_macro']:.3f}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_result(result: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Métriques écrites : {path}")
    return path


def log_result_to_mlflow(result: dict, prefix: str, artifact_dir: Path) -> None:
    """Logge métriques + rapport JSON + matrice de confusion dans le run MLflow actif."""
    import mlflow

    mlflow.log_metrics({f"{prefix}_{k}": v for k, v in result["overall"].items()})
    for media, metrics in result["by_media"].items():
        mlflow.log_metrics({f"{prefix}_{media}_{k}": v for k, v in metrics.items()})

    json_path = save_result(result, artifact_dir / f"{prefix}_metrics.json")
    png_path = plot_confusion_matrix(result, artifact_dir / f"{prefix}_confusion_matrix.png")
    mlflow.log_artifact(str(json_path))
    mlflow.log_artifact(str(png_path))


def evaluate_model_dir(
    model_dir: Path,
    data_dir: Path = Path(cfg.paths.processed_dir),
    split: str = "test",
    use_meta: bool | None = None,
    run_name: str | None = None,
    neutral_meta: bool = False,
) -> dict:
    """
    Ré-évalue un modèle sauvegardé et logge le résultat comme run MLflow.
    Sert notamment à vérifier le modèle historique V6 avec le code actuel.

    neutral_meta : remplace sentiment/ironie par « neutre / non ironique » pour tous les textes,
    c'est-à-dire les conditions d'usage réelles (le public ne connaît pas ces annotations).
    """
    import mlflow

    from sherlock.model.classifier import load_model
    from sherlock.tracking import git_commit, setup_mlflow

    use_meta = cfg.model.use_meta if use_meta is None else use_meta
    run_name = run_name or f"eval_{model_dir.parent.name}_{model_dir.name}"
    df = pd.read_parquet(data_dir / f"{split}.parquet")
    if neutral_meta:
        df = df.assign(sentiment="neutre", ironie=False)

    model, tokenizer = load_model(model_dir)
    model.to(get_device())

    setup_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {"git_commit": git_commit(), "stage": "evaluation", "model_dir": str(model_dir)}
        )
        mlflow.log_params(
            {
                "model_dir": str(model_dir),
                "split": split,
                "use_meta": use_meta,
                "neutral_meta": neutral_meta,
            }
        )
        result = evaluate_dataframe(model, tokenizer, df, use_meta)
        result |= {"run_id": run.info.run_id, "model_dir": str(model_dir), "use_meta": use_meta}
        log_result_to_mlflow(result, split, Path(cfg.paths.reports_dir) / "metrics" / run_name)

    return result["overall"]
