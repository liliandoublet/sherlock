"""
clean - Nettoyage et normalisation des textes politiques.
"""

from sherlock.clean.regex_rules import clean_text, count_words_excluding_token
from sherlock.clean.language import is_french
from sherlock.clean.pipeline import run as clean_pipeline

__all__ = ["clean_text", "count_words_excluding_token", "is_french", "clean_pipeline"]