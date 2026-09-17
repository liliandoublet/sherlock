# Sherlock - reproduire tout le projet
#   make all            données + ré-évaluation V6 + baseline (pas de fine-tuning)
#   make train          fine-tuning CamemBERT (~30-60 min sur GPU)
#   make report         régénère le tableau de résultats du README

RUN := uv run
SHERLOCK := $(RUN) sherlock

.PHONY: help install data eval-legacy baselines train train-ablation report all \
        mlflow demo test lint format docker-build

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-16s %s\n", $$1, $$2}'

install: ## Installe toutes les dépendances (torch CUDA inclus)
	uv sync --all-extras

data: ## Importe les splits historiques (data/raw/legacy/*.csv -> data/processed/*.parquet)
	$(SHERLOCK) dataset import-legacy data/raw/legacy

eval-legacy: ## Ré-évalue le modèle historique V6 (models/legacy/camembert_v6_meta) dans MLflow
	$(SHERLOCK) evaluate models/legacy/camembert_v6_meta --run-name legacy_v6_meta

baselines: ## Baseline TF-IDF + LogReg dans MLflow
	$(SHERLOCK) baselines

train: ## Fine-tuning CamemBERT, texte + sentiment/ironie (run MLflow camembert_meta)
	$(SHERLOCK) train --run-name camembert_meta --output-dir models/camembert_party

train-ablation: ## Fine-tuning CamemBERT, texte seul (run MLflow camembert_nometa)
	$(SHERLOCK) train --no-meta --run-name camembert_nometa --output-dir models/camembert_nometa

report: ## Régénère le tableau de résultats du README depuis reports/
	$(SHERLOCK) report

all: data eval-legacy baselines report ## Tout sauf le fine-tuning

mlflow: ## Interface MLflow sur http://localhost:5000
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

demo: ## Démo Streamlit sur http://localhost:8501
	$(RUN) streamlit run app/streamlit_app.py

test: ## Tests unitaires + couverture
	$(RUN) pytest

lint: ## Vérifie le style (ruff)
	$(RUN) ruff check src tests app
	$(RUN) ruff format --check src tests app

format: ## Corrige le style (ruff)
	$(RUN) ruff check --fix src tests app
	$(RUN) ruff format src tests app

docker-build: ## Construit l'image de la démo (même image que la CD)
	docker build -t sherlock-demo .
