"""
Mini-CamemBERT et tokenizer word-level pour les tests : aucun téléchargement, quelques secondes
sur CPU. À importer après `pytest.importorskip("torch")` / `("transformers")`.
"""

from pathlib import Path

import transformers
from tokenizers import Tokenizer, models, pre_tokenizers

PARTIES = ["EELV", "LFI", "LR", "PCF", "PS", "Reconquête", "Renaissance", "RN"]
WORDS = ["texte", "du", "parti", "[NOIRONY]", "[SENT_neutre]", "[SENT_positif]", "[IRONY]"]


def tiny_tokenizer():
    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
    for i, word in enumerate(WORDS, start=4):
        vocab[word] = i
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>"
    )


def tiny_model(parties=PARTIES):
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


def save_tiny_model(directory: Path) -> Path:
    """Écrit un modèle + tokenizer minuscules au format save_pretrained."""
    directory.mkdir(parents=True, exist_ok=True)
    tiny_model().save_pretrained(directory)
    tiny_tokenizer().save_pretrained(directory)
    return directory
