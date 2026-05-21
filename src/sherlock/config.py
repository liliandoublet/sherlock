"""
config.py
---------
Charge params.yaml et expose la configuration via des modèles Pydantic.
Utilisation :
    from sherlock.config import cfg
    print(cfg.parties)   # ['EELV', 'LFI', ...]
    print(cfg.io.separator)  # '|'
"""

from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field


# ── Sous-modèles ──────────────────────────────────────────────────────────────

class ProjectConfig(BaseModel):
    name: str
    version: str
    language: str

class PathsConfig(BaseModel):
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path
    logs_dir: Path

class IOConfig(BaseModel):
    separator: str
    encoding: str

class CleaningConfig(BaseModel):
    min_words: int
    min_words_strict: int
    chunk_size: int
    mention_token: str

class BalanceConfig(BaseModel):
    per_party: int
    seed: int

class SplitConfig(BaseModel):
    train: float
    val: float
    test: float
    seed: int

class ModelConfig(BaseModel):
    name: str
    max_length: int
    batch_size: int
    learning_rate: float
    epochs: int
    warmup_ratio: float
    early_stopping_patience: int


# ── Modèle principal ──────────────────────────────────────────────────────────

class Config(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    io: IOConfig
    parties: list[str]
    cleaning: CleaningConfig
    balance: BalanceConfig
    split: SplitConfig
    model: ModelConfig


# ── Chargement ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_config(path: str = "params.yaml") -> Config:
    """
    Charge et valide params.yaml.
    Le @lru_cache garantit qu'on ne lit le fichier qu'une seule fois
    même si on appelle load_config() depuis plusieurs modules.
    """
    params_path = Path(path)
    if not params_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {params_path.resolve()}\n"
            "Make sure you run sherlock from the project root directory."
        )
    with params_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


# ── Singleton pratique ────────────────────────────────────────────────────────
# Permet d'écrire `from sherlock.config import cfg` directement
# sans appeler load_config() à chaque fois.
cfg: Config = load_config()
