import torch.nn as nn
from loguru import logger
from transformers import AutoModel

from sherlock.config import cfg


class PartyClassifier(nn.Module):
    """
    CamemBERTa-v2 fine-tuné pour la classification de parti politique.

    Architecture :
        CamemBERTa-v2 (frozen ou trainable)
        -> pooling [CLS]
        -> dropout
        -> linear(hidden_size, n_classes)
    """

    def __init__(self, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.n_classes = n_classes
        self.encoder = AutoModel.from_pretrained(cfg.model.name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

        logger.info(
            f"PartyClassifier initialisé : {cfg.model.name} "
            f"-> {n_classes} classes, hidden_size={hidden_size}"
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # token [CLS]
        dropped = self.dropout(cls_output)
        logits = self.classifier(dropped)
        return logits
