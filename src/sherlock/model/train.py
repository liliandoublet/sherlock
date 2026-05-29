import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from sherlock.config import cfg
from sherlock.model.classifier import PartyClassifier
from sherlock.model.tokenizer import get_tokenizer

# ── Dataset PyTorch ───────────────────────────────────────────────────────────


class PartyDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.model.max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ── Fonctions utilitaires ─────────────────────────────────────────────────────


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les splits parquet depuis data/processed/."""
    train = pd.read_parquet(data_dir / "train.parquet")
    val = pd.read_parquet(data_dir / "val.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    logger.info(f"Splits chargés : {len(train)} train / {len(val)} val / {len(test)} test")
    return train, val, test


def evaluate(model, loader, device) -> dict:
    """Évalue le modèle sur un DataLoader."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    return {
        "loss": round(total_loss / len(loader), 4),
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "f1_macro": round(f1_score(all_labels, all_preds, average="macro"), 4),
        "f1_weighted": round(f1_score(all_labels, all_preds, average="weighted"), 4),
    }


# ── Boucle d'entraînement principale ─────────────────────────────────────────


def train(
    data_dir: Path = Path(cfg.paths.processed_dir),
    output_dir: Path = Path(cfg.paths.models_dir) / "camembert_party",
    run_name: str = "camembert_party",
):
    """
    Fine-tune CamemBERTa-v2 pour la classification de parti.
    Toutes les métriques sont loggées dans MLflow.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")

    # ── Chargement des données ────────────────────────────────────────────────
    train_df, val_df, test_df = load_splits(data_dir)

    # Encodage des labels
    le = LabelEncoder()
    le.fit(cfg.parties)
    train_labels = le.transform(train_df["parti"].tolist())
    val_labels = le.transform(val_df["parti"].tolist())
    test_labels = le.transform(test_df["parti"].tolist())
    n_classes = len(le.classes_)
    logger.info(f"Classes : {list(le.classes_)}")

    # ── Tokenisation ──────────────────────────────────────────────────────────
    tokenizer = get_tokenizer()
    train_ds = PartyDataset(train_df["texte"].tolist(), train_labels.tolist(), tokenizer)
    val_ds = PartyDataset(val_df["texte"].tolist(), val_labels.tolist(), tokenizer)
    test_ds = PartyDataset(test_df["texte"].tolist(), test_labels.tolist(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=cfg.model.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.model.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.model.batch_size)

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = PartyClassifier(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.model.learning_rate,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * cfg.model.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.model.warmup_ratio),
        num_training_steps=total_steps,
    )

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_experiment("sherlock-party-classification")

    with mlflow.start_run(run_name=run_name):
        # Log hyperparamètres
        mlflow.log_params(
            {
                "model": cfg.model.name,
                "epochs": cfg.model.epochs,
                "batch_size": cfg.model.batch_size,
                "learning_rate": cfg.model.learning_rate,
                "max_length": cfg.model.max_length,
                "warmup_ratio": cfg.model.warmup_ratio,
                "n_classes": n_classes,
                "train_size": len(train_ds),
                "val_size": len(val_ds),
                "device": str(device),
            }
        )

        best_val_f1 = 0.0
        patience_count = 0

        # ── Boucle epochs ─────────────────────────────────────────────────────
        for epoch in range(1, cfg.model.epochs + 1):
            model.train()
            train_loss = 0.0
            start = time.time()

            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.model.epochs}"):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            # Métriques epoch
            avg_train_loss = round(train_loss / len(train_loader), 4)
            val_metrics = evaluate(model, val_loader, device)
            elapsed = round(time.time() - start, 1)

            logger.info(
                f"Epoch {epoch} | train_loss={avg_train_loss} | "
                f"val_loss={val_metrics['loss']} | "
                f"val_acc={val_metrics['accuracy']} | "
                f"val_f1={val_metrics['f1_macro']} | "
                f"{elapsed}s"
            )

            # Log MLflow
            mlflow.log_metrics(
                {
                    "train_loss": avg_train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "val_f1_weighted": val_metrics["f1_weighted"],
                },
                step=epoch,
            )

            # Early stopping + sauvegarde meilleur modèle
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                patience_count = 0
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                logger.info(f"Meilleur modèle sauvegardé (f1={best_val_f1})")
            else:
                patience_count += 1
                if patience_count >= cfg.model.early_stopping_patience:
                    logger.info(f"Early stopping à l'epoch {epoch}")
                    break

        # ── Évaluation finale sur test ─────────────────────────────────────────
        logger.info("Évaluation finale sur test set...")
        model.load_state_dict(torch.load(output_dir / "best_model.pt"))
        test_metrics = evaluate(model, test_loader, device)

        mlflow.log_metrics(
            {
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_f1_weighted": test_metrics["f1_weighted"],
            }
        )

        logger.info(
            f"Test final : acc={test_metrics['accuracy']} | f1_macro={test_metrics['f1_macro']}"
        )

        # Log modèle dans MLflow
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_dict(
            {str(i): label for i, label in enumerate(le.classes_)},
            "label_mapping.json",
        )

    return test_metrics
