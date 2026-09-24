# CI/CD (GitHub Actions)

## CI : à chaque push et pull request sur `main`

[`ci.yml`](../.github/workflows/ci.yml) installe le projet dans un environnement propre, puis lance
`ruff check`, `ruff format --check` et `pytest`. Torch est installé en version CPU, ce qui permet aux
tests du modèle de s'exécuter réellement : entraînement complet d'un CamemBERT miniature (MLflow
compris) et parcours de l'appli Streamlit avec `AppTest`.

## CD : à chaque tag `vX.Y.Z`

[`release.yml`](../.github/workflows/release.yml) :

1. relance toute la CI (pas de livraison sans tests verts) ;
2. vérifie que le tag correspond à la version de `pyproject.toml` ;
3. publie une **GitHub Release** : wheel, sdist et archive des métriques ;
4. construit et publie l'**image Docker** de la démo sur `ghcr.io/liliandoublet/sherlock`.

À quoi ça sert : chaque version devient un livrable figé, testé et récupérable sans rien installer.

```bash
docker run -p 8501:8501 -v sherlock_hf:/root/.cache/huggingface ghcr.io/liliandoublet/sherlock:latest
```

## Publier une version

```bash
# 1. incrémenter `version` dans pyproject.toml et params.yaml, puis commit + push
git tag v0.3.0
git push origin v0.3.0
```

Ensuite, onglet *Actions* : le workflow *Release* doit passer au vert. Pour que n'importe qui puisse
tirer l'image, rendre le paquet public (GitHub → profil → *Packages* → `sherlock` → *Package settings*
→ *Change visibility*).

## En local

```bash
make lint test       # ce que fait la CI
make docker-build    # construit la même image
make docker-run
```
