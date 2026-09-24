"""
Tests pour le module model/
On mocke les modèles HuggingFace pour ne pas télécharger de poids.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

PARTIES = ["EELV", "LFI", "LR", "PCF", "PS", "Reconquête", "Renaissance", "RN"]


class FakeModel(torch.nn.Module):
    """Modèle minimal : renvoie des logits favorisant une classe fixe."""

    def __init__(self, n_classes: int = 8, winner: int = 0):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.winner = winner
        labels = sorted(PARTIES)[:n_classes]
        self.config = MagicMock(num_labels=n_classes, id2label=dict(enumerate(labels)))

    def forward(self, input_ids, attention_mask, **kwargs):
        logits = torch.zeros(input_ids.shape[0], self.config.num_labels)
        logits[:, self.winner] = 5.0
        return MagicMock(logits=logits)


def fake_tokenizer(texts, **kwargs):
    n = len(texts)
    enc = {
        "input_ids": torch.zeros(n, 4, dtype=torch.long),
        "attention_mask": torch.ones(n, 4, dtype=torch.long),
    }
    batch = MagicMock()
    batch.to.return_value = enc
    return batch


# ── features ──────────────────────────────────────────────────────────────────


def test_meta_prefix_matches_legacy_format():
    """Le préfixe reproduit le format du modèle historique V6."""
    from sherlock.model.features import meta_prefix

    assert meta_prefix("négatif", False) == "[NOIRONY] [SENT_négatif] "
    assert meta_prefix(" Positif ", True) == "[IRONY] [SENT_positif] "


def test_build_inputs_with_and_without_meta():
    from sherlock.model.features import build_inputs

    df = pd.DataFrame({"texte": ["abc"], "sentiment": ["neutre"], "ironie": [True]})
    assert build_inputs(df, use_meta=False) == ["abc"]
    assert build_inputs(df, use_meta=True) == ["[IRONY] [SENT_neutre] abc"]


# ── classifier ────────────────────────────────────────────────────────────────


def test_label_maps_sorted_like_label_encoder():
    """Même ordre que sklearn.LabelEncoder utilisé par les modèles historiques."""
    from sherlock.model.classifier import label_maps

    id2label, label2id = label_maps(PARTIES)
    assert id2label[0] == "EELV"
    assert id2label[7] == "Renaissance"
    assert all(label2id[label] == i for i, label in id2label.items())


def test_build_model_passes_label_maps():
    import sherlock.model.classifier as clf

    with patch.object(clf.AutoModelForSequenceClassification, "from_pretrained") as mock_load:
        clf.build_model(PARTIES, model_name="fake")

    kwargs = mock_load.call_args.kwargs
    assert kwargs["num_labels"] == 8
    assert kwargs["label2id"]["RN"] == 5


def test_load_model_requires_config(tmp_path):
    from sherlock.model.classifier import load_model

    with pytest.raises(FileNotFoundError):
        load_model(tmp_path)


def test_load_model_missing_path_or_hub_id_is_a_clear_error():
    """Ni dossier local ni dépôt Hub : message français, pas de trace brute de transformers."""
    from sherlock.model.classifier import load_model

    with pytest.raises(FileNotFoundError, match="Modèle introuvable"):
        load_model("/chemin/absolu/qui/nexiste/pas")


def test_load_model_accepts_str_and_path(tmp_path):
    pytest.importorskip("transformers")
    from sherlock.model.classifier import load_model
    from tests.unit.tiny_model import save_tiny_model

    directory = save_tiny_model(tmp_path / "model")
    for source in (directory, str(directory)):
        model, tokenizer = load_model(source)
        assert model.config.num_labels == 8
        assert tokenizer is not None


# ── evaluate ──────────────────────────────────────────────────────────────────


def test_predict_proba_keeps_input_order():
    """Le tri par longueur ne doit pas mélanger les prédictions."""
    from sherlock.model.evaluate import predict_proba

    probs = predict_proba(FakeModel(winner=2), fake_tokenizer, ["long texte", "a", "moyen"])
    assert probs.shape == (3, 8)
    assert (probs.argmax(axis=1) == 2).all()
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])


def test_evaluate_dataframe_metrics_and_media():
    from sherlock.model.evaluate import evaluate_dataframe

    df = pd.DataFrame(
        {
            "texte": ["a", "b", "c", "d"],
            "parti": ["EELV", "EELV", "LFI", "LFI"],
            "media": ["Twitter", "site_web", "Twitter", "site_web"],
            "sentiment": ["neutre"] * 4,
            "ironie": [False] * 4,
        }
    )
    result = evaluate_dataframe(FakeModel(winner=0), fake_tokenizer, df, use_meta=True)

    assert result["n"] == 4
    assert result["overall"]["accuracy"] == 0.5
    assert set(result["by_media"]) == {"Twitter", "site_web"}
    assert len(result["confusion_matrix"]) == 8


# ── predict ───────────────────────────────────────────────────────────────────


def test_predictor_output_structure():
    import sherlock.model.predict as predict_module

    with patch.object(
        predict_module, "load_model", return_value=(FakeModel(winner=5), fake_tokenizer)
    ):
        predictor = predict_module.Predictor(model_dir=MagicMock())
        results = predictor.predict(["texte"], ["positif"], [False])

    assert len(results) == 1
    assert results[0]["parti"] == "RN"
    assert set(results[0]["all_scores"]) == set(PARTIES)
    assert 0 < results[0]["confidence"] <= 1
