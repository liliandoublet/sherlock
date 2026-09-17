# 🔍 Sherlock

> Classification du parti politique d'un texte français par fine-tuning de CamemBERT

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![CI](https://github.com/liliandoublet/sherlock/actions/workflows/ci.yml/badge.svg)](https://github.com/liliandoublet/sherlock/actions/workflows/ci.yml)
[![Release](https://github.com/liliandoublet/sherlock/actions/workflows/release.yml/badge.svg)](https://github.com/liliandoublet/sherlock/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sherlock attribue un texte politique français (tweet ou extrait de site officiel) à l'un de
8 partis : `EELV` `LFI` `LR` `PCF` `PS` `Reconquête` `Renaissance` `RN`.

Le modèle est `camembert-base` fine-tuné sur 14 696 textes annotés. Les expériences sont
suivies dans **MLflow**, le code est testé et livré par un pipeline **CI/CD GitHub Actions**,
et une **démo Streamlit** permet de le tester en direct.

---

## En bref : chaque affirmation et sa preuve

| Affirmation | Où le vérifier |
|---|---|
| Fine-tuning de CamemBERT | [`src/sherlock/model/train.py`](src/sherlock/model/train.py), `make train` |
| F1 macro ≈ 0,63 sur le test set (8 classes équilibrées) | [Résultats](#résultats), [`reports/metrics/`](reports/metrics/), [`reports/legacy/`](reports/legacy/) |
| +5,7 points face à une baseline TF-IDF + LogReg | [Résultats](#résultats) |
| Suivi des expériences avec MLflow | [Suivi MLflow](#suivi-des-expériences-mlflow), `make mlflow` |
| CI : lint + tests à chaque push | [`.github/workflows/ci.yml`](.github/workflows/ci.yml), badge CI |
| CD : release + image Docker à chaque tag | [`.github/workflows/release.yml`](.github/workflows/release.yml), onglet *Releases* et *Packages* |
| Démo en direct | [Démo](#démo-streamlit), `make demo` |

---

## Résultats

Test set de 1 470 textes (≈ 184 par parti). Le hasard donne une F1 macro de 0,125.
Ce tableau est **généré** par `make report` à partir des fichiers de métriques : aucun
chiffre n'est saisi à la main.

<!-- results:start -->
**Runs du pipeline actuel** (test set, 1 470 textes, suivis dans MLflow) :

| Modèle | Run | Accuracy | F1 macro | F1 Twitter | F1 site web |
|---|---|---|---|---|---|
| CamemBERT fine-tuné, texte + sentiment/ironie | `camembert_meta` | *non lancé (`make train`)* | | | |
| CamemBERT fine-tuné, texte seul (ablation) | `camembert_nometa` | *non lancé (`make train-ablation`)* | | | |
| Modèle historique V6 ré-évalué avec le code actuel | `legacy_v6_meta` | 0,626 | **0,627** | 0,594 | 0,730 |
| Baseline TF-IDF + régression logistique | `tfidf_logreg` | 0,570 | **0,570** | 0,555 | 0,615 |

**Historique du projet de fin d'études** (même test set, transformers 4.52, exports CSV dans `reports/legacy/`) :

| Version | F1 macro | F1 Twitter | F1 site web | Note |
|---|---|---|---|---|
| V2 | **0,638** | 0,602 | 0,750 |  |
| V3 | **0,634** | 0,600 | 0,742 |  |
| V4 | **0,627** | 0,592 | 0,739 | dossier nommé « sans sentiment/ironie » mais évalué avec (cf. décision 008) |
| V5 | **0,557** | 0,521 | 0,651 |  |
| V6 | **0,629** | 0,597 | 0,728 | CamemBERT + préfixes sentiment/ironie, modèle ré-évalué ci-dessus |
| V8 | **0,646** | — | — | CamemBERT + sentiment/ironie en features numériques |
| V9 | **0,541** | — | — | EuroBERT (pas CamemBERT) |
<!-- results:end -->

**Lecture des résultats**

- **Le 0,63 est reproduit.** Le modèle V6, ré-évalué avec le code actuel, obtient 0,627.
  L'export d'origine indiquait 0,629. Modèle, données et logique de préfixes sont identiques ;
  le seul changement est la version de `transformers` (4.52 → 5.9).
- **CamemBERT bat la baseline de 5,7 points** (0,627 contre 0,570). La baseline retombe
  exactement sur sa valeur historique (0,5695).
- **Les tweets sont plus difficiles** que les extraits de sites officiels (≈ 0,59 contre ≈ 0,73) :
  textes plus courts (41 mots médians contre 103) et plus réactifs à l'actualité.
- **Confusions entre partis voisins** (V6, test set) : LR ↔ RN (53 erreurs), PCF ↔ PS (36),
  LR ↔ Renaissance (36), RN ↔ Reconquête (35), LFI ↔ PCF (35). Reconquête est le parti le mieux
  reconnu (F1 0,73), RN le moins bien (0,57). Voir les matrices de confusion dans `reports/`.
- **Apport du sentiment et de l'ironie : inconnu à ce jour.** L'ancien projet n'avait pas
  d'ablation fiable (cf. [décision 008](docs/decisions.md)). `make train-ablation` la produit.

---

## Tout reproduire

### Prérequis

- [uv](https://docs.astral.sh/uv/) et Python 3.12
- Un GPU NVIDIA pour le fine-tuning (testé sur RTX 3060 Ti 8 Go). L'évaluation et la démo tournent aussi sur CPU.
- Les fichiers non versionnés (corpus de tweets et poids trop lourds pour git) :

  | Fichier | Emplacement attendu |
  |---|---|
  | Splits historiques | `data/raw/legacy/{train,val,test}.csv` (séparateur `\|`) |
  | Modèle V6 (optionnel, pour `eval-legacy` et la démo) | `models/legacy/camembert_v6_meta/` |

### Avec le Makefile

```bash
make install          # uv sync --all-extras
make all              # données + ré-évaluation V6 + baseline + tableau du README (quelques minutes)
make train            # fine-tuning CamemBERT, texte + sentiment/ironie (~30-60 min GPU)
make train-ablation   # fine-tuning CamemBERT, texte seul
make report           # met à jour le tableau de résultats ci-dessus
make mlflow           # http://localhost:5000
make demo             # http://localhost:8501
make help             # toutes les cibles
```

### Commandes équivalentes

```bash
uv sync --all-extras

uv run sherlock dataset import-legacy data/raw/legacy          # CSV -> data/processed/*.parquet validés
uv run sherlock evaluate models/legacy/camembert_v6_meta --run-name legacy_v6_meta
uv run sherlock baselines
uv run sherlock train --run-name camembert_meta --output-dir models/camembert_party
uv run sherlock train --no-meta --run-name camembert_nometa --output-dir models/camembert_nometa
uv run sherlock report

uv run sherlock predict "Il faut sortir du nucléaire et investir dans les renouvelables" \
  --model-dir models/camembert_party --sentiment positif
```

---

## Suivi des expériences (MLflow)

Toutes les commandes `train`, `evaluate` et `baselines` créent un run dans l'expérience
`sherlock-party-classification` (backend `sqlite:///mlflow.db`, configuré dans `params.yaml`).

| Logué | Détail |
|---|---|
| Paramètres | modèle, epochs, batch, learning rate, seed, fp16, `use_meta`, tailles des splits, device |
| Tags | commit git, étape (`training` / `evaluation` / `baseline`), dossier du modèle |
| Métriques par epoch | `train_loss`, `val_accuracy`, `val_f1_macro`, `val_f1_weighted`, durée |
| Métriques finales | `test_*` globales et par média (`test_Twitter_f1_macro`, `test_site_web_f1_macro`…) |
| Artefacts | `params.yaml`, rapport de classification JSON, matrice de confusion PNG |

```bash
make mlflow   # puis http://localhost:5000 → comparer les runs, tracer val_f1_macro par epoch
```

Les métriques de test sont aussi écrites dans `reports/metrics/<run>/` (versionné), pour rester
consultables sans MLflow.

---

## CI/CD (GitHub Actions)

**CI** ([`ci.yml`](.github/workflows/ci.yml)) : à chaque push et pull request sur `main` :
`ruff check`, `ruff format --check`, puis `pytest`. Torch est installé en version CPU, ce qui
permet aux tests du modèle de s'exécuter réellement, y compris un test de bout en bout de la
boucle d'entraînement sur un CamemBERT miniature.

**CD** ([`release.yml`](.github/workflows/release.yml)) : à chaque tag `vX.Y.Z` :

1. relance toute la CI (pas de livraison sans tests verts) ;
2. vérifie que le tag correspond à la version de `pyproject.toml` ;
3. publie une **GitHub Release** : wheel + sdist + archive des métriques ;
4. construit et publie l'**image Docker** de la démo sur `ghcr.io/liliandoublet/sherlock`.

**À quoi ça sert ?** Chaque version devient un livrable figé, testé et récupérable sans rien
installer. N'importe qui (recruteur, collègue, serveur) peut lancer exactement la version
publiée :

```bash
docker run -p 8501:8501 -v "$(pwd)/models:/app/models" ghcr.io/liliandoublet/sherlock:latest
```

Publier une version :

```bash
git tag v0.2.0 && git push origin v0.2.0
```

---

## Démo Streamlit

```bash
make demo   # http://localhost:8501
```

- **Prédire** : saisir un texte, choisir sentiment et ironie, afficher les probabilités par parti.
  Le bouton *Exemple du test set* tire un vrai texte jamais vu à l'entraînement et affiche le parti
  réel à côté de la prédiction.
- **Résultats** : tableau des runs, F1 par parti et par média, matrice de confusion.
- Le modèle se choisit dans la barre latérale parmi les dossiers présents dans `models/`
  (fine-tuning actuel, V6 historique, ablation).
- `.streamlit/config.toml` désactive l'auto-reload (`fileWatcherType = "none"`) : Streamlit
  et `transformers` ont un conflit connu qui pollue le terminal de tracebacks `torchvision`
  sinon. Après une modification de `app/streamlit_app.py`, relancer `make demo` manuellement.

---

## Données

| | |
|---|---|
| Volume | 14 696 textes : 11 756 train / 1 470 val / 1 470 test |
| Classes | 8 partis, équilibrés (≈ 1 837 textes par parti) |
| Sources | 11 136 tweets de 24 comptes de personnalités politiques, 3 560 extraits de sites officiels des partis |
| Période | septembre 2019 → mai 2025 |
| Annotations | sentiment (`positif` / `neutre` / `négatif`) et ironie (booléen) par Gemini 2.0 Flash, température 0 |
| Répartition | 50,8 % négatif, 27,6 % positif, 21,6 % neutre ; 13,2 % ironique |

**Limites connues**

- 8 textes identiques entre train et test (7 train/val, 1 val/test). Les splits sont conservés
  tels quels pour rester comparables aux résultats historiques ; `import-legacy` les signale.
- Les étiquettes sentiment et ironie viennent d'un LLM, sans validation humaine systématique.
- 5 coquilles `positre` corrigées en `positif` à l'import (aucune dans le test set).
- Le label est le parti de l'auteur, pas le contenu idéologique du texte : un tweet purement
  factuel (« je serai l'invité politique du 8h30 ») ne porte aucun signal partisan.

---

## Modèle

- `almanach/camembert-base` + tête de classification linéaire (`AutoModelForSequenceClassification`)
- Entrée : `[NOIRONY] [SENT_négatif] texte…`, tronquée à 512 tokens (option `--no-meta` : texte seul)
- AdamW, lr 2e-5, batch 16, 4 epochs, fp16, seed 92, sélection du meilleur epoch sur la F1 macro de validation
- Hyperparamètres dans [`params.yaml`](params.yaml), identiques à ceux du modèle historique V6

---

## Structure du projet

```
sherlock/
├── src/sherlock/
│   ├── collect/        # Scrapers (base abstraite, Wikipedia)
│   ├── clean/          # Nettoyage (regex, détection de langue)
│   ├── annotate/       # Annotation sentiment / ironie
│   ├── dataset/
│   │   ├── legacy.py   # Import et validation des splits historiques
│   │   ├── merge.py    # Fusion multi-sources
│   │   ├── balance.py  # Équilibrage par parti
│   │   └── split.py    # Split stratifié
│   ├── model/
│   │   ├── features.py    # Préfixes sentiment / ironie
│   │   ├── classifier.py  # Construction / chargement du modèle
│   │   ├── train.py       # Fine-tuning + MLflow
│   │   ├── evaluate.py    # Métriques, matrice de confusion, ré-évaluation
│   │   ├── baselines.py   # TF-IDF + LogReg
│   │   └── predict.py     # Inférence
│   ├── report.py       # Génération du tableau de résultats
│   ├── tracking.py     # Configuration MLflow
│   ├── config.py       # params.yaml validé par Pydantic
│   └── cli.py          # CLI Typer
├── app/streamlit_app.py
├── tests/unit/
├── reports/
│   ├── metrics/        # Métriques des runs actuels (versionné)
│   └── legacy/         # Exports de l'ancien projet (versionné)
├── data/               # Non versionné
├── models/             # Non versionné
├── docs/decisions.md
├── .github/workflows/  # ci.yml, release.yml
├── Dockerfile
├── Makefile
└── params.yaml
```

Les modules `collect`, `clean` et `annotate` servent à reconstruire un corpus à partir de zéro ;
ils ne sont pas utilisés pour les résultats ci-dessus, qui reposent sur les splits historiques.

---

## Décisions techniques

Voir [docs/decisions.md](docs/decisions.md) : choix de CamemBERT-base, conservation des splits
historiques, rôle des métadonnées sentiment/ironie, parquet, uv…

---

## Développement

```bash
make test     # pytest + couverture
make lint     # ruff check + ruff format --check
make format   # corrections automatiques
```

---

## Auteur

**Lilian Doublet**
Master Data Analytics, Intelligence et Sécurité — Rennes School of Business

## License

MIT — voir [LICENSE](LICENSE)
