from pathlib import Path

import mlflow
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from loguru import logger

from sherlock.config import cfg
from sherlock.model.tokenizer import get_tokenizer
from sherlock.model.classifier import PartyClassifier
from sherlock.model.train import PartyDataset, evaluate


def full_evaluation(
    model_path: Path,
    data_dir:   Path = Path(cfg.paths.processed_dir),
) -> dict:
    """
    Évalue un modèle sauvegardé sur le test set.
    Affiche le rapport de classification complet.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chargement du test set
    test_df = pd.read_parquet(data_dir / "test.parquet")
    le      = LabelEncoder()
    le.fit(cfg.parties)
    test_labels = le.transform(test_df["parti"].tolist())

    # Modèle
    tokenizer = get_tokenizer()
    model     = PartyClassifier(n_classes=len(le.classes_)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    test_ds     = PartyDataset(test_df["texte"].tolist(), test_labels.tolist(), tokenizer)
    test_loader = DataLoader(test_ds, batch_size=cfg.model.batch_size)

    # Prédictions
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())

    # Rapport
    report = classification_report(
        all_labels, all_preds,
        target_names=le.classes_,
        output_dict=True,
    )
    logger.info("\n" + classification_report(all_labels, all_preds, target_names=le.classes_))

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"Matrice de confusion :\n{cm}")

    return report