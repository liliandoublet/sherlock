from functools import lru_cache
from transformers import AutoTokenizer
from loguru import logger
from sherlock.config import cfg


@lru_cache(maxsize=1)
def get_tokenizer():
    """
    Charge le tokenizer une seule fois (LRU cache).
    Utilise le modèle défini dans params.yaml.
    """
    logger.info(f"Chargement tokenizer : {cfg.model.name}")
    return AutoTokenizer.from_pretrained(cfg.model.name)


def tokenize_batch(texts: list[str], tokenizer=None) -> dict:
    """
    Tokenise une liste de textes.

    Returns:
        Dict avec input_ids, attention_mask, token_type_ids.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=cfg.model.max_length,
        return_tensors="pt",
    )