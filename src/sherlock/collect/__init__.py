"""
collect - Scrapers pour les différentes sources de données.
"""

from sherlock.collect.base import BaseScraper
from sherlock.collect.wikipedia import WikipediaScraper

__all__ = ["BaseScraper", "WikipediaScraper"] 