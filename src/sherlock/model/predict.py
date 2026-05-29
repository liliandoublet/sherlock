from pathlib import Path

import torch
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from sherlock.config import cfg
from sherlock.model.classifier import PartyClassifier
from sherlock.model.tokenizer import get_tokenizer


def predict_text(
    text: str,
    model_path: Path = Path(cfg.paths.models_dir) / "camembert_party" / "best_model.pt",
) -> dict:
    """
    Prédit le parti politique d'un texte.

    Returns:
        Dict avec 'parti', 'confidence', et 'all_scores'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()
    le = LabelEncoder()
    le.fit(cfg.parties)

    model = PartyClassifier(n_classes=len(le.classes_)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=cfg.model.max_length,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        logits = model(
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device),
        )
        probs = torch.softmax(logits, dim=-1).squeeze().cpu()
        pred_idx = probs.argmax().item()
        confidence = round(probs[pred_idx].item(), 4)

    all_scores = {label: round(probs[i].item(), 4) for i, label in enumerate(le.classes_)}

    result = {
        "parti": le.classes_[pred_idx],
        "confidence": confidence,
        "all_scores": all_scores,
    }
    logger.info(f"Prédiction : {result['parti']} ({confidence:.1%})")
    return result
