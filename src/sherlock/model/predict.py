"""
predict.py
----------
Inférence sur un ou plusieurs textes à partir d'un modèle sauvegardé.
"""

from pathlib import Path

from loguru import logger

from sherlock.config import cfg
from sherlock.model.classifier import load_model
from sherlock.model.evaluate import get_device, predict_proba
from sherlock.model.features import meta_prefix

DEFAULT_MODEL_DIR = Path(cfg.paths.models_dir) / "camembert_party"


class Predictor:
    """Charge le modèle une fois, prédit ensuite à la demande (CLI, Streamlit)."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, use_meta: bool | None = None):
        self.model, self.tokenizer = load_model(model_dir)
        self.model.to(get_device())
        self.use_meta = cfg.model.use_meta if use_meta is None else use_meta
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def predict(
        self,
        texts: list[str],
        sentiments: list[str] | None = None,
        ironies: list[bool] | None = None,
        use_meta: bool | None = None,
    ) -> list[dict]:
        if self.use_meta if use_meta is None else use_meta:
            sentiments = sentiments or ["neutre"] * len(texts)
            ironies = ironies or [False] * len(texts)
            inputs = [
                meta_prefix(s, i) + t for t, s, i in zip(texts, sentiments, ironies, strict=True)
            ]
        else:
            inputs = list(texts)

        probs = predict_proba(self.model, self.tokenizer, inputs)
        results = []
        for row in probs:
            scores = {label: float(p) for label, p in zip(self.labels, row, strict=True)}
            best = max(scores, key=scores.get)
            results.append({"parti": best, "confidence": scores[best], "all_scores": scores})
        return results


def predict_text(
    text: str,
    model_dir: Path = DEFAULT_MODEL_DIR,
    sentiment: str = "neutre",
    irony: bool = False,
) -> dict:
    """Prédit le parti politique d'un texte. Retourne parti, confidence, all_scores."""
    result = Predictor(model_dir).predict([text], [sentiment], [irony])[0]
    logger.info(f"Prédiction : {result['parti']} ({result['confidence']:.1%})")
    return result
