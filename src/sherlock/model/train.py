"""
train.py
--------
Fine-tuning de CamemBERT pour la classification de parti, suivi dans MLflow.

Chaque run logge : hyperparamètres, commit git, loss/F1 par epoch (train et val),
métriques finales sur test (globales et par média), rapport de classification,
matrice de confusion et params.yaml.
"""

import random
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from sherlock.config import cfg
from sherlock.model.classifier import build_model, label_maps, load_model
from sherlock.model.evaluate import evaluate_dataframe, get_device, log_result_to_mlflow
from sherlock.model.features import build_inputs
from sherlock.model.tokenizer import get_tokenizer
from sherlock.tracking import git_commit, setup_mlflow

# ── Utilitaires ───────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les splits parquet depuis data/processed/."""
    train = pd.read_parquet(data_dir / "train.parquet")
    val = pd.read_parquet(data_dir / "val.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    logger.info(f"Splits chargés : {len(train)} train / {len(val)} val / {len(test)} test")
    return train, val, test


def encode(df: pd.DataFrame, tokenizer, label2id: dict[str, int], use_meta: bool) -> list[dict]:
    """Tokenise sans padding (le padding est fait par batch par le collator)."""
    enc = tokenizer(
        build_inputs(df, use_meta),
        truncation=True,
        max_length=cfg.model.max_length,
    )
    labels = df["parti"].map(label2id).tolist()
    return [
        {"input_ids": ids, "attention_mask": mask, "labels": label}
        for ids, mask, label in zip(enc["input_ids"], enc["attention_mask"], labels, strict=True)
    ]


# ── Boucle d'entraînement principale ─────────────────────────────────────────


def train(
    data_dir: Path = Path(cfg.paths.processed_dir),
    output_dir: Path = Path(cfg.paths.models_dir) / "camembert_party",
    run_name: str = "camembert_party",
    use_meta: bool | None = None,
    epochs: int | None = None,
) -> dict:
    """
    Fine-tune CamemBERT et retourne les métriques test.
    Le meilleur modèle (F1 macro sur val) est sauvegardé dans output_dir.
    """
    use_meta = cfg.model.use_meta if use_meta is None else use_meta
    epochs = epochs or cfg.model.epochs
    set_seed(cfg.model.seed)
    device = get_device()
    use_amp = cfg.model.fp16 and device.type == "cuda"
    logger.info(f"Device : {device} | fp16={use_amp} | use_meta={use_meta}")

    # ── Données ───────────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_splits(data_dir)
    _, label2id = label_maps(cfg.parties)

    tokenizer = get_tokenizer()
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        encode(train_df, tokenizer, label2id, use_meta),
        batch_size=cfg.model.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(cfg.model.seed),
    )

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = build_model(cfg.parties).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.model.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── MLflow run ────────────────────────────────────────────────────────────
    setup_mlflow()
    artifact_dir = Path(cfg.paths.reports_dir) / "metrics" / run_name

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"git_commit": git_commit(), "stage": "training"})
        mlflow.log_params(
            {
                "model": cfg.model.name,
                "use_meta": use_meta,
                "epochs": epochs,
                "batch_size": cfg.model.batch_size,
                "learning_rate": cfg.model.learning_rate,
                "weight_decay": cfg.model.weight_decay,
                "warmup_ratio": cfg.model.warmup_ratio,
                "max_length": cfg.model.max_length,
                "seed": cfg.model.seed,
                "fp16": use_amp,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "device": str(device),
            }
        )
        mlflow.log_artifact("params.yaml")

        best_val_f1 = -1.0
        patience_count = 0
        step = 0

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            start = time.time()

            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = model(**batch).loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item()
                step += 1
                if step % 50 == 0:
                    mlflow.log_metric("train_loss_step", loss.item(), step=step)

            val = evaluate_dataframe(model, tokenizer, val_df, use_meta)["overall"]
            avg_train_loss = train_loss / len(train_loader)
            elapsed = time.time() - start
            logger.info(
                f"Epoch {epoch} | train_loss={avg_train_loss:.4f} | "
                f"val_acc={val['accuracy']:.4f} | val_f1={val['f1_macro']:.4f} | {elapsed:.0f}s"
            )
            mlflow.log_metrics(
                {
                    "train_loss": avg_train_loss,
                    "val_accuracy": val["accuracy"],
                    "val_f1_macro": val["f1_macro"],
                    "val_f1_weighted": val["f1_weighted"],
                    "epoch_seconds": elapsed,
                },
                step=epoch,
            )

            # Early stopping + sauvegarde du meilleur modèle (F1 macro val)
            if val["f1_macro"] > best_val_f1:
                best_val_f1 = val["f1_macro"]
                patience_count = 0
                output_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                mlflow.log_metric("best_epoch", epoch)
                logger.info(f"Meilleur modèle sauvegardé (val f1={best_val_f1:.4f})")
            else:
                patience_count += 1
                if patience_count >= cfg.model.early_stopping_patience:
                    logger.info(f"Early stopping à l'epoch {epoch}")
                    break

        # ── Évaluation finale sur test avec le meilleur modèle ───────────────
        logger.info("Évaluation finale sur test set...")
        del model
        torch.cuda.empty_cache()
        best_model, _ = load_model(output_dir)
        best_model.to(device)

        test_result = evaluate_dataframe(best_model, tokenizer, test_df, use_meta)
        test_result["run_id"] = run.info.run_id
        test_result["model_dir"] = str(output_dir)
        test_result["use_meta"] = use_meta
        mlflow.log_metric("best_val_f1_macro", best_val_f1)
        log_result_to_mlflow(test_result, "test", artifact_dir)
        mlflow.set_tag("model_dir", str(output_dir))

    return test_result["overall"]
