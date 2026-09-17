from functools import lru_cache

from loguru import logger
from transformers import AutoTokenizer

from sherlock.config import cfg


@lru_cache(maxsize=1)
def get_tokenizer():
    """
    Charge le tokenizer une seule fois (LRU cache).
    Utilise le modèle défini dans params.yaml.
    """
    logger.info(f"Chargement tokenizer : {cfg.model.name}")
    return AutoTokenizer.from_pretrained(cfg.model.name)
