"""
space.py
--------
Publie la démo Streamlit en ligne, sur un Space Hugging Face (SDK Docker).

Le Space reçoit une copie minimale du dépôt (code, appli, métriques) plus space/Dockerfile et
space/README.md, qui deviennent son Dockerfile et sa fiche. Le modèle n'est pas copié : le Space le
télécharge depuis le Hub au build (demo.model dans params.yaml).
"""

import shutil
from pathlib import Path

from sherlock.config import cfg

ROOT = Path(__file__).resolve().parents[2]

# Ce dont l'image a besoin, relatif à la racine du dépôt
PROJECT_FILES = ["pyproject.toml", "params.yaml", "src", "app", "reports", ".streamlit"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def stage_space(build_dir: Path, model: str | None = None, root: Path = ROOT) -> Path:
    """Prépare dans build_dir exactement les fichiers que recevra le Space."""
    model = model or cfg.demo.model
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    for name in PROJECT_FILES:
        source = root / name
        if source.is_dir():
            shutil.copytree(source, build_dir / name, ignore=IGNORE)
        else:
            shutil.copy2(source, build_dir / name)
    shutil.copy2(root / "space" / "Dockerfile", build_dir / "Dockerfile")
    card = (root / "space" / "README.md").read_text(encoding="utf-8")
    (build_dir / "README.md").write_text(card.replace("__MODEL__", model), encoding="utf-8")
    return build_dir


def publish_space(space: str, build_dir: Path, private: bool = False) -> str:
    """Crée le Space s'il n'existe pas, puis remplace tout son contenu par build_dir."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=build_dir,
        repo_id=space,
        repo_type="space",
        delete_patterns="*",
        commit_message=f"Démo Sherlock {cfg.project.version}",
    )
    return f"https://huggingface.co/spaces/{space}"
