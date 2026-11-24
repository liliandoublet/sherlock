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

## Site vitrine : GitHub Pages

[`pages.yml`](../.github/workflows/pages.yml) publie le dossier `site/` à chaque push sur `main` qui le
modifie. À activer une seule fois : *Settings* → *Pages* → *Source* : **GitHub Actions**. Le site est
ensuite servi sur <https://liliandoublet.github.io/sherlock/>.

La page est un seul fichier HTML, sans build ni dépendance. Ses données (métriques, exemples) sont
dans la balise `<script id="sherlock-data">`, réécrite par `make site` depuis `reports/metrics/` :
aucun chiffre n'y est saisi à la main, et un test vérifie qu'elle est à jour.

```bash
make site          # met à jour site/index.html (réponses aux exemples si le modèle V6 local est là)
make site-serve    # aperçu sur http://localhost:8000
```

Avec le modèle local `models/legacy/camembert_v6_meta`, `make site` calcule aussi les probabilités
réelles des six exemples de la démo, affichées sur le site. Sans modèle, la page invite à les coller
dans la démo et les réponses déjà calculées sont conservées.

## En local

```bash
make lint test       # ce que fait la CI
make docker-build    # construit la même image
make docker-run
```
