"""
Tests pour le module config.py
"""

import pytest
from sherlock.config import load_config, Config


def test_config_loads():
    """Le fichier params.yaml se charge sans erreur."""
    cfg = load_config()
    assert isinstance(cfg, Config)


def test_config_parties():
    """Les 8 partis politiques sont présents."""
    cfg = load_config()
    expected = {"EELV", "LFI", "LR", "PCF", "PS", "Reconquête", "Renaissance", "RN"}
    assert set(cfg.parties) == expected


def test_config_split_sums_to_one():
    """Les ratios train/val/test doivent sommer à 1.0."""
    cfg = load_config()
    total = cfg.split.train + cfg.split.val + cfg.split.test
    assert abs(total - 1.0) < 1e-9, f"Split ratios sum to {total}, expected 1.0"


def test_config_io():
    """Le séparateur CSV est bien le pipe."""
    cfg = load_config()
    assert cfg.io.separator == "|"


def test_config_paths_are_path_objects():
    """Les chemins sont bien des objets Path, pas des strings."""
    from pathlib import Path
    cfg = load_config()
    assert isinstance(cfg.paths.data_dir, Path)
    assert isinstance(cfg.paths.raw_dir, Path)
