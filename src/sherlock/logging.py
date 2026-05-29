"""
logging.py
----------
Configure loguru pour le projet.
Utilisation :
    from sherlock.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Lecture du CSV...")
    logger.warning("Fichier manquant : {path}", path=p)
    logger.error("Erreur inattendue")
"""

import sys
from pathlib import Path

from loguru import logger as _logger


def setup_logging(log_dir: str | Path = "logs", level: str = "INFO") -> None:
    """
    Configure loguru : console colorée + fichier rotatif.
    À appeler une seule fois au démarrage (dans cli.py ou main.py).
    """
    _logger.remove()  # supprime le handler par défaut

    # Console : colorée, lisible
    _logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Fichier : rotatif, 10 Mo max, garde 7 jours
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    _logger.add(
        log_path / "sherlock_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
    )


def get_logger(name: str):
    """Retourne un logger bindé au nom du module appelant."""
    return _logger.bind(name=name)
