# 🔍 Sherlock

> Détection automatique d'idéologie politique dans les textes français

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![CI](https://github.com/liliandoublet/sherlock/actions/workflows/ci.yml/badge.svg)](https://github.com/liliandoublet/sherlock/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-74%25-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sherlock est un pipeline NLP complet qui classifie l'idéologie politique de textes français (tweets, communiqués de partis) parmi 8 partis politiques, avec détection auxiliaire de sentiment et d'ironie.

---

## Partis supportés

`EELV` `LFI` `LR` `PCF` `PS` `Reconquête` `Renaissance` `RN`

---

## Architecture du pipeline

```
┌─────────────────────────────────────────────────────────┐
│                      COLLECTE                           │
│   Twitter (tweets)  │  Sites web  │  Wikipedia (neutre) │
└────────────────────────────┬────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────┐
│                     NETTOYAGE                           │
│        Regex (Twitter)  │  Gemini async (Web)           │
└────────────────────────────┬────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────┐
│                    ANNOTATION                           │
│     Sentiment (CamemBERT)  │  Ironie (CamemBERT)        │
└────────────────────────────┬────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────┐
│                     DATASET                             │
│         Merge  │  Balance (1392/parti)  │  Split        │
└────────────────────────────┬────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────┐
│                  FINE-TUNING                            │
│        CamemBERTa-v2  +  MLflow tracking                │
└────────────────────────────┬────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────┐
│                  PRÉDICTION                             │
│            sherlock predict "texte..."                  │
└─────────────────────────────────────────────────────────┘
```

---

## Quickstart

```bash
git clone https://github.com/liliandoublet/sherlock.git
cd sherlock

# Installer les dépendances
uv sync --all-extras

# Configurer les secrets
cp .env.example .env
# Éditer .env et ajouter GEMINI_API_KEY

# Vérifier l'installation
sherlock --help
```

---

## Utilisation

### Voir la configuration courante

```bash
sherlock info
```

### Nettoyer un CSV de tweets

```bash
sherlock clean data/raw/twitter/RN.csv data/interim/RN_clean.csv --source twitter
sherlock clean data/raw/web/PS.csv data/interim/PS_clean.csv --source web
```

### Annoter sentiment et ironie

```bash
sherlock annotate data/interim/merged.csv data/interim/annotated.csv
```

### Construire le dataset

```bash
# Fusionner les sources
sherlock dataset merge \
  data/interim/twitter.csv \
  data/interim/web.csv \
  --output data/interim/merged.csv

# Équilibrer par parti
sherlock dataset balance \
  data/interim/merged.csv \
  data/interim/balanced.csv \
  --per-party 1392

# Créer les splits train/val/test
sherlock dataset split data/interim/balanced.csv
```

### Entraîner le modèle

```bash
sherlock train --run-name "camembert_v1"

# Visualiser les expériences
mlflow ui
# Ouvrir http://localhost:5000
```

### Prédire sur un texte

```bash
sherlock predict "La transition écologique est notre priorité absolue"
sherlock predict "Nous devons réduire l'immigration illégale"
```

---

## Structure du projet

```
sherlock/
│
├── src/sherlock/
│   ├── collect/          # Scrapers (Twitter, Web, Wikipedia)
│   │   ├── base.py       # Classe abstraite BaseScraper
│   │   └── wikipedia.py  # Scraper Wikipedia (classe neutre)
│   │
│   ├── clean/            # Nettoyage textuel
│   │   ├── regex_rules.py  # Règles déterministes (URLs, mentions...)
│   │   ├── language.py     # Détection de langue (FR)
│   │   └── pipeline.py     # Pipeline complet
│   │
│   ├── annotate/         # Annotation automatique
│   │   ├── sentiment.py    # CamemBERT sentiment (AlloCine)
│   │   ├── irony.py        # CamemBERT ironie
│   │   └── pipeline.py     # Pipeline combiné
│   │
│   ├── dataset/          # Construction du dataset
│   │   ├── merge.py        # Fusion multi-sources
│   │   ├── balance.py      # Équilibrage par parti
│   │   └── split.py        # Split stratifié 80/10/10
│   │
│   ├── model/            # Fine-tuning et inférence
│   │   ├── classifier.py   # PartyClassifier (CamemBERTa-v2)
│   │   ├── tokenizer.py    # Tokenizer avec cache LRU
│   │   ├── train.py        # Boucle d'entraînement + MLflow
│   │   ├── evaluate.py     # Évaluation détaillée
│   │   └── predict.py      # Inférence single text
│   │
│   ├── config.py         # Configuration Pydantic (params.yaml)
│   ├── logging.py        # Loguru configuré
│   └── cli.py            # CLI Typer unifiée
│
├── tests/
│   ├── unit/             # 47 tests unitaires
│   └── fixtures/         # Données de test
│
├── data/                 # Versionné par DVC (ignoré par Git)
│   ├── raw/
│   ├── interim/
│   └── processed/        # train.parquet, val.parquet, test.parquet
│
├── models/               # Modèles entraînés (ignoré par Git)
├── docs/                 # Documentation technique
├── notebooks/            # Analyse exploratoire
├── params.yaml           # Configuration centralisée
├── pyproject.toml        # Dépendances et outils
└── .env.example          # Template des secrets
```

---

## Stack technique

| Composant | Technologie | Raison du choix |
|---|---|---|
| Modèle NLP | CamemBERTa-v2 | Architecture DeBERTa, meilleur que BERT sur classification |
| Nettoyage LLM | Gemini 2.0 Flash (async) | Compréhension contextuelle pour communiqués complexes |
| Sentiment | tblard/tf-allocine | CamemBERT fine-tuné sur corpus français |
| Ironie | mrm8488/camembert-base-finetuned-irony | Spécialisé français |
| ML tracking | MLflow | Gratuit, local, visualisation des runs |
| Package manager | uv | 10-100× plus rapide que pip, lock file strict |
| Config | Pydantic + YAML | Validation typée, source unique de vérité |
| Tests | pytest (47 tests, 74%) | Mocks HuggingFace, pas de téléchargement en CI |
| Logs | loguru | Structurés, rotatifs, colorés |

---

## Données

### Sources

| Source | Type | Partis | Volume cible |
|---|---|---|---|
| Twitter/X | Tweets (280 car.) | Tous | 1392/parti |
| Sites officiels | Communiqués (long) | Tous | Équilibré |
| Wikipedia | Articles neutres | aucun | 300 articles |

### Schéma commun

| Colonne | Type | Description |
|---|---|---|
| `texte` | str | Texte nettoyé |
| `compte` | str | Compte ou titre source |
| `parti` | str | Label politique |
| `date` | str | Date (YYYY-MM-DD) |
| `source` | str | URL d'origine |
| `media` | str | `twitter`, `site_web`, `wikipedia` |

### Après annotation

| Colonne | Valeurs | Description |
|---|---|---|
| `sentiment` | `POSITIVE`, `NEGATIVE` | Polarité du texte |
| `sentiment_score` | 0.0 à 1.0 | Confiance du modèle |
| `ironie` | `ironic`, `not_ironic` | Présence d'ironie |
| `ironie_score` | 0.0 à 1.0 | Confiance du modèle |

---

## Résultats

| Modèle | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|
| CamemBERTa-v2 fine-tuné | *NAN* | *NAN* | *NAN* |
| Baseline zero-shot LLM | *NAN* | *NAN* | *NAN* |

---

## Décisions techniques

Les choix d'architecture sont documentés dans [docs/decisions.md](docs/decisions.md).

Exemples de décisions documentées :
- Pourquoi CamemBERTa-v2 plutôt que CamemBERT-base
- Pourquoi Gemini pour le web et regex pour Twitter
- Pourquoi under-sampling plutôt que class weights
- Pourquoi parquet plutôt que CSV pour les splits

---

## Développement

### Lancer les tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ -v --cov=sherlock --cov-report=html
```

### Linter et formatter

```bash
uv run ruff check src/
uv run ruff format src/
```

### Ajouter une dépendance

```bash
# Dépendance core
uv add nom-du-paquet

# Dépendance optionnelle (groupe data, ml, dev...)
uv add --optional data nom-du-paquet
```

---

## Variables d'environnement

Copier `.env.example` vers `.env` et remplir :

| Variable | Obligatoire | Description |
|---|---|---|
| `GEMINI_API_KEY` | Oui (étape web) | Clé API Google Gemini |
| `ANTHROPIC_API_KEY` | Non | Clé API Anthropic (baselines) |
| `MLFLOW_TRACKING_URI` | Non | URI MLflow (défaut : `file:./mlruns`) |
| `HF_TOKEN` | Non | Token HuggingFace pour upload de modèles |


## Auteur

**Lilian Doublet**
Master Data Analytics, Intelligence et Sécurité — Rennes School of Business

---

## License

MIT — voir [LICENSE](LICENSE)
