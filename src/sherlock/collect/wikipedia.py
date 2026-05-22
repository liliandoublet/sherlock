import re
import time
import random
from pathlib import Path

import pandas as pd
import requests

from sherlock.collect.base import BaseScraper


# ── Constantes ────────────────────────────────────────────────────────────────

API_BASE  = "https://fr.wikipedia.org/w/api.php"
SPACE_RE  = re.compile(r"\s+")
UA_POOL   = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
]


class WikipediaScraper(BaseScraper):
    """
    Scrape des articles Wikipedia FR comme corpus neutre
    (parti = 'aucun', media = 'wikipedia').
    """

    def __init__(self):
        super().__init__(parti="aucun")
        self.session = requests.Session()

    def _api(self, params: dict) -> dict:
        params["format"] = "json"
        r = self.session.get(
            API_BASE,
            params=params,
            timeout=20,
            headers={"User-Agent": random.choice(UA_POOL)},
        )
        r.raise_for_status()
        return r.json()

    def _random_titles(self, n: int) -> list[str]:
        data = self._api({
            "action": "query",
            "list": "random",
            "rnnamespace": "0",
            "rnlimit": min(n, 500),
        })
        return [item["title"] for item in data["query"]["random"]]

    def _fetch_article(self, title: str) -> dict | None:
        data = self._api({
            "action": "query",
            "prop": "extracts|info|revisions",
            "titles": title,
            "explaintext": 1,
            "exsectionformat": "plain",
            "rvprop": "timestamp",
            "inprop": "url",
        })
        page = next(iter(data["query"]["pages"].values()))
        if "missing" in page:
            return None

        text = SPACE_RE.sub(" ", page.get("extract", "").replace("\xa0", " ")).strip()
        if len(text) < 200:
            return None

        return {
            "texte":  text,
            "compte": page["title"],
            "parti":  "aucun",
            "date":   page.get("revisions", [{}])[0].get("timestamp", ""),
            "source": page.get("fullurl", ""),
            "media":  "wikipedia",
        }

    def fetch(self, n: int = 300, delay: float = 1.0) -> pd.DataFrame:
        """
        Scrape n articles Wikipedia aléatoires.

        Args:
            n:      nombre d'articles cibles
            delay:  pause entre requêtes (politesse)

        Returns:
            DataFrame avec colonnes SCHEMA.
        """
        self.logger.info(f"Scraping Wikipedia : {n} articles cibles")
        titles = self._random_titles(n * 2)
        rows = []

        for title in titles:
            if len(rows) >= n:
                break
            try:
                art = self._fetch_article(title)
                if art:
                    rows.append(art)
                time.sleep(delay)
            except Exception as e:
                self.logger.warning(f"Erreur sur '{title}' : {e}")

        df = pd.DataFrame(rows)
        self.logger.info(f"{len(df)} articles récupérés")
        return self.validate(df)