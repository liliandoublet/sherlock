"""
Test de bout en bout de la boucle d'entraînement, sans télécharger CamemBERT.

On construit un CamemBERT minuscule (1 couche, hidden 16) et un tokenizer
word-level en local : train() tourne réellement (optimisation, sauvegarde,
rechargement, évaluation, MLflow) en quelques secondes sur CPU.
"""

from unittest.mock import patch

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
mlflow = pytest.importorskip("mlflow")
transformers = pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers

PARTIES = ["EELV", "LFI", "LR", "PCF", "PS", "Reconquête", "Renaissance", "RN"]


def tiny_tokenizer():
    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
    for i, word in enumerate(["texte", "du", "parti", "[NOIRONY]", "[SENT_neutre]"], start=4):
        vocab[word] = i
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>"
    )


def tiny_model(parties):
    from sherlock.model.classifier import label_maps

    id2label, label2id = label_maps(parties)
    config = transformers.CamembertConfig(
        vocab_size=16,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=520,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    return transformers.CamembertForSequenceClassification(config)


def write_splits(data_dir):
    rows = [
        {
            "texte": f"texte du parti {party}",
            "parti": party,
            "media": media,
            "sentiment": "neutre",
            "ironie": False,
        }
        for party in PARTIES
        for media in ("Twitter", "site_web")
    ]
    data_dir.mkdir()
    for name in ("train", "val", "test"):
        pd.DataFrame(rows).to_parquet(data_dir / f"{name}.parquet", index=False)


def test_train_end_to_end(tmp_path, monkeypatch):
    import sherlock.model.train as train_module
    from sherlock.config import cfg

    data_dir = tmp_path / "processed"
    write_splits(data_dir)
    monkeypatch.setattr(cfg.paths, "reports_dir", tmp_path / "reports")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    def fake_setup_mlflow():
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("test")

    with (
        patch.object(train_module, "get_tokenizer", return_value=tiny_tokenizer()),
        patch.object(train_module, "build_model", side_effect=tiny_model),
        patch.object(train_module, "get_device", return_value=torch.device("cpu")),
        patch.object(train_module, "setup_mlflow", side_effect=fake_setup_mlflow),
    ):
        metrics = train_module.train(
            data_dir=data_dir,
            output_dir=tmp_path / "model",
            run_name="smoke",
            epochs=1,
        )

    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert (tmp_path / "model" / "config.json").exists()
    assert (tmp_path / "reports" / "metrics" / "smoke" / "test_metrics.json").exists()

    run = mlflow.search_runs(
        experiment_names=["test"], output_format="list", filter_string="run_name = 'smoke'"
    )[0]
    assert run.data.params["use_meta"] == "True"
    assert "val_f1_macro" in run.data.metrics
    assert "test_f1_macro" in run.data.metrics
