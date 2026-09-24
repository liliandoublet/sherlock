# 🔍 Sherlock

> Quel parti politique français se cache derrière ce texte ?

[![CI](https://github.com/liliandoublet/sherlock/actions/workflows/ci.yml/badge.svg)](https://github.com/liliandoublet/sherlock/actions/workflows/ci.yml)
[![Release](https://github.com/liliandoublet/sherlock/actions/workflows/release.yml/badge.svg)](https://github.com/liliandoublet/sherlock/actions/workflows/release.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sherlock est un CamemBERT fine-tuné qui attribue un texte politique français (tweet, extrait de site
officiel) à l'un de 8 partis : `EELV` `LFI` `LR` `PCF` `PS` `Reconquête` `Renaissance` `RN`.
Il atteint une **F1 macro de 0,63** sur des textes jamais vus, contre 0,57 pour une baseline TF-IDF.

## Essayer la démo

**Avec Docker** (le plus simple, rien à installer) :

```bash
docker run -p 8501:8501 -v sherlock_hf:/root/.cache/huggingface ghcr.io/liliandoublet/sherlock:latest
```

**Sans Docker** (avec [uv](https://docs.astral.sh/uv/)) :

```bash
git clone https://github.com/liliandoublet/sherlock && cd sherlock
uv run --extra demo streamlit run app/streamlit_app.py
```

Puis ouvrir <http://localhost:8501>. Au premier lancement, le modèle (~420 Mo) est téléchargé depuis
Hugging Face. L'installation locale télécharge PyTorch (plusieurs Go) ; l'image Docker, en version CPU, est plus légère.

## Résultats

Test set de 1 470 textes jamais vus, 8 partis équilibrés (le hasard donne 12,5 %). Tableau généré par
`make report` à partir de `reports/metrics/` ; historique complet dans [docs/results.md](docs/results.md).

<!-- results:start -->
| Modèle | F1 macro | Accuracy | F1 tweets | F1 sites officiels |
|---|---|---|---|---|
| Sherlock (V6), conditions de la démo | **0,626** | 0,624 | 0,591 | 0,733 |
| Sherlock (V6) avec sentiment/ironie annotés | **0,627** | 0,626 | 0,594 | 0,730 |
| Baseline TF-IDF + régression logistique | **0,570** | 0,570 | 0,555 | 0,615 |
<!-- results:end -->

Les deux premières lignes montrent que, pour ce modèle, sentiment et ironie changent à peine le
résultat à l'inférence : la démo n'en a pas besoin. Les tweets sont plus difficiles que les extraits
de sites officiels (textes beaucoup plus courts).

## Comment ça marche

- **Données** : 14 696 textes annotés (tweets de 24 comptes, sites officiels, 2019-2025), non redistribués. → [docs/data.md](docs/data.md)
- **Modèle** : `camembert-base` fine-tuné (AdamW, lr 2e-5, 4 epochs, fp16), hyperparamètres dans [`params.yaml`](params.yaml).
- **MLflow** : chaque entraînement et chaque évaluation enregistre paramètres, courbes par epoch, métriques par média et commit git. → [docs/reproduce.md](docs/reproduce.md#suivi-des-expériences-mlflow)
- **CI/CD** : GitHub Actions lance lint et tests à chaque push ; à chaque tag `vX.Y.Z`, une Release et l'image Docker de la démo sont publiées. → [docs/ci-cd.md](docs/ci-cd.md)
- **Tests** : unitaires, dont un entraînement complet sur un mini-CamemBERT et l'appli Streamlit.

## Reproduire

```bash
make install    # dépendances (GPU NVIDIA recommandé pour l'entraînement)
make all        # données, évaluation du modèle historique V6, baseline, tableaux
make train      # fine-tuning CamemBERT (~10 min sur RTX 3060 Ti)
make mlflow     # suivi des expériences : http://localhost:5000
```

Fichiers requis, commandes détaillées et dépannage : [docs/reproduce.md](docs/reproduce.md).

## Limites

- Le modèle prédit le parti **de l'auteur**, pas la position idéologique du texte, et se trompe régulièrement, surtout entre partis proches (LR/RN, PS/PCF, RN/Reconquête).
- Corpus limité à 24 comptes et aux sites officiels : il généralise mal aux autres registres. Sentiment et ironie ont été annotés par un LLM, sans validation humaine systématique.
- Les opinions politiques sont une donnée sensible : ne pas utiliser cet outil pour profiler quelqu'un.

Choix techniques : [docs/decisions.md](docs/decisions.md).

## Auteur

**Lilian Doublet**, Master Data Analytics, Intelligence et Sécurité, Rennes School of Business. Licence MIT.
