"""
Tests pour le module model/
On mocke les modèles HuggingFace pour ne pas télécharger de poids.
"""

from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")


def test_party_classifier_forward():
    """PartyClassifier produit le bon nombre de logits."""
    mock_encoder = MagicMock()
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.zeros(2, 10, 768)
    mock_encoder.return_value = mock_output
    mock_encoder.config.hidden_size = 768

    with patch("sherlock.model.classifier.AutoModel.from_pretrained", return_value=mock_encoder):
        from importlib import reload

        import sherlock.model.classifier as clf_module

        reload(clf_module)
        model = clf_module.PartyClassifier(n_classes=8)
        ids = torch.zeros(2, 10, dtype=torch.long)
        mask = torch.ones(2, 10, dtype=torch.long)
        logits = model(ids, mask)

    assert logits.shape == (2, 8)


def test_party_classifier_n_classes():
    """PartyClassifier accepte n'importe quel nombre de classes."""
    mock_encoder = MagicMock()
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.zeros(1, 5, 512)
    mock_encoder.return_value = mock_output
    mock_encoder.config.hidden_size = 512

    with patch("sherlock.model.classifier.AutoModel.from_pretrained", return_value=mock_encoder):
        from importlib import reload

        import sherlock.model.classifier as clf_module

        reload(clf_module)
        model = clf_module.PartyClassifier(n_classes=3)
        assert model.n_classes == 3


def test_predict_text_output_structure():
    """predict_text retourne un dict avec les bonnes clés."""
    # On mocke tout ce qui touche au modèle et aux poids
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }

    # Instance mockée qui retourne un vrai tensor
    mock_instance = MagicMock(spec=["eval", "load_state_dict", "to", "__call__"])
    mock_instance.to.return_value = mock_instance
    mock_instance.return_value = torch.randn(1, 8)
    mock_instance.load_state_dict = MagicMock(return_value=None)

    with (
        patch("sherlock.model.predict.get_tokenizer", return_value=mock_tokenizer),
        patch("sherlock.model.predict.PartyClassifier", return_value=mock_instance),
        patch("sherlock.model.predict.torch.load", return_value={}),
        patch("torch.nn.Module.load_state_dict", return_value=None),
    ):
        from importlib import reload

        import sherlock.model.predict as predict_module

        reload(predict_module)
        result = predict_module.predict_text("test text", model_path=MagicMock())

    assert "parti" in result
    assert "confidence" in result
    assert "all_scores" in result
    assert isinstance(result["all_scores"], dict)
