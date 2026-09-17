"""
baselines.py
------------
Baseline TF-IDF + régression logistique, sur les mêmes splits que CamemBERT.
Mêmes réglages que le script historique (results_baselines.py) : F1 macro 0,570.
"""

import time
from pathlib import Path

import joblib
import mlflow
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

from sherlock.config import cfg
from sherlock.model.classifier import label_maps
from sherlock.model.evaluate import compute_metrics, plot_confusion_matrix, save_result
from sherlock.model.train import load_splits
from sherlock.tracking import git_commit, setup_mlflow


def train_tfidf_logreg(
    data_dir: Path = Path(cfg.paths.processed_dir),
    output_dir: Path = Path(cfg.paths.models_dir) / "baselines",
    run_name: str = "tfidf_logreg",
) -> dict:
    train_df, _, test_df = load_splits(data_dir)
    id2label, label2id = label_maps(cfg.parties)
    labels = [id2label[i] for i in range(len(id2label))]
    y_train = train_df["parti"].map(label2id)
    y_test = test_df["parti"].map(label2id)

    params = {
        "max_features": 50_000,
        "ngram_range": (1, 2),
        "max_iter": 300,
        "class_weight": "balanced",
        "seed": cfg.model.seed,
    }
    pipeline = make_pipeline(
        TfidfVectorizer(max_features=params["max_features"], ngram_range=params["ngram_range"]),
        LogisticRegression(
            max_iter=params["max_iter"],
            class_weight=params["class_weight"],
            random_state=params["seed"],
        ),
    )

    setup_mlflow()
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"git_commit": git_commit(), "stage": "baseline"})
        mlflow.log_params({"model": "tfidf_logreg", **params})

        start = time.time()
        pipeline.fit(train_df["texte"], y_train)
        y_pred = pipeline.predict(test_df["texte"])
        mlflow.log_metric("fit_seconds", time.time() - start)

        result = {
            "n": len(test_df),
            "overall": compute_metrics(y_test, y_pred),
            "by_media": {},
            "report": classification_report(
                y_test, y_pred, labels=range(len(labels)), target_names=labels, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(
                y_test, y_pred, labels=range(len(labels))
            ).tolist(),
            "labels": labels,
        }
        for media, idx in test_df.groupby("media").groups.items():
            pos = test_df.index.get_indexer(idx)
            result["by_media"][media] = compute_metrics(y_test.iloc[pos], y_pred[pos])

        mlflow.log_metrics({f"test_{k}": v for k, v in result["overall"].items()})
        artifact_dir = Path(cfg.paths.reports_dir) / "metrics" / run_name
        mlflow.log_artifact(str(save_result(result, artifact_dir / "test_metrics.json")))
        mlflow.log_artifact(
            str(plot_confusion_matrix(result, artifact_dir / "test_confusion_matrix.png"))
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, output_dir / f"{run_name}.joblib")

    logger.info(f"TF-IDF + LogReg : f1_macro={result['overall']['f1_macro']:.4f}")
    return result["overall"]
