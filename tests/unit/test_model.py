import pytest
import torch
from unittest.mock import patch, MagicMock


def test_party_classifier_forward():
    """PartyClassifier produit le bon nombre de logits."""
    mock_encoder = MagicMock()
    mock_output  = MagicMock()
    mock_output.last_hidden_state = torch.zeros(2, 10, 768)
    mock_encoder.return_value     = mock_output
    mock_encoder.config.hidden_size = 768

    with patch("sherlock.model.classifier.AutoModel.from_pretrained", return_value=mock_encoder):
        from sherlock.model.classifier import PartyClassifier
        model  = PartyClassifier(n_classes=8)
        ids    = torch.zeros(2, 10, dtype=torch.long)
        mask   = torch.ones(2, 10, dtype=torch.long)
        logits = model(ids, mask)

    assert logits.shape == (2, 8)


def test_party_classifier_n_classes():
    """PartyClassifier accepte n'importe quel nombre de classes."""
    mock_encoder = MagicMock()
    mock_output  = MagicMock()
    mock_output.last_hidden_state = torch.zeros(1, 5, 512)
    mock_encoder.return_value     = mock_output
    mock_encoder.config.hidden_size = 512

    with patch("sherlock.model.classifier.AutoModel.from_pretrained", return_value=mock_encoder):
        from importlib import reload
        import sherlock.model.classifier as clf_module
        reload(clf_module)
        model = clf_module.PartyClassifier(n_classes=3)
        assert model.n_classes == 3


def test_predict_text_output_structure():
    """predict_text retourne un dict avec les bonnes clés."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids":      torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(1, 8)
    mock_model.eval = MagicMock()

    with patch("sherlock.model.predict.get_tokenizer", return_value=mock_tokenizer), \
         patch("sherlock.model.predict.PartyClassifier", return_value=mock_model), \
         patch("sherlock.model.predict.torch.load", return_value={}), \
         patch("sherlock.model.predict.Path.exists", return_value=True):
        from sherlock.model.predict import predict_text
        result = predict_text("test text", model_path=MagicMock())

    assert "parti" in result
    assert "confidence" in result
    assert "all_scores" in result
    assert isinstance(result["all_scores"], dict)