# Sherlock - reproduire tout le projet
#   make all            données + ré-évaluation V6 + baseline (pas de fine-tuning)
#   make train          fine-tuning CamemBERT (~30-60 min sur GPU)
#   make report         régénère le tableau de résultats du README
#   make demo           démo publique (modèle téléchargé depuis Hugging Face)

RUN := uv run
SHERLOCK := $(RUN) sherlock

.PHONY: help install install-demo data eval-legacy baselines train train-ablation report all \
        mlflow demo demo-local publish-model test lint format docker-build docker-run

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-16s %s\n", $$1, $$2}'

install: ## Installe toutes les dépendances (torch CUDA inclus)
	uv sync --all-extras

install-demo: ## Installe uniquement ce qu'il faut pour la démo
	uv sync --extra demo

data: ## Importe les splits historiques (data/raw/legacy/*.csv -> data/processed/*.parquet)
	$(SHERLOCK) dataset import-legacy data/raw/legacy

eval-legacy: ## Ré-évalue le modèle historique V6 dans MLflow (avec annotations, puis conditions de la démo)
	$(SHERLOCK) evaluate models/legacy/camembert_v6_meta --run-name legacy_v6_meta
	$(SHERLOCK) evaluate models/legacy/camembert_v6_meta --neutral-meta --run-name legacy_v6_public_input

baselines: ## Baseline TF-IDF + LogReg dans MLflow
	$(SHERLOCK) baselines

train: ## Fine-tuning CamemBERT, texte + sentiment/ironie (run MLflow camembert_meta)
	$(SHERLOCK) train --run-name camembert_meta --output-dir models/camembert_party

train-ablation: ## Fine-tuning CamemBERT, texte seul (run MLflow camembert_nometa)
	$(SHERLOCK) train --no-meta --run-name camembert_nometa --output-dir models/camembert_nometa

report: ## Régénère le tableau du README et docs/results.md depuis reports/
	$(SHERLOCK) report

all: data eval-legacy baselines report ## Tout sauf le fine-tuning

mlflow: ## Interface MLflow sur http://localhost:5000
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

demo: ## Démo publique sur http://localhost:8501 (modèle téléchargé depuis Hugging Face)
	$(RUN) --extra demo streamlit run app/streamlit_app.py

demo-local: ## Démo avec le modèle local models/legacy/camembert_v6_meta
	SHERLOCK_MODEL=models/legacy/camembert_v6_meta $(RUN) --extra demo streamlit run app/streamlit_app.py

publish-model: ## Publie le modèle V6 sur Hugging Face (nécessite `hf auth login`)
	$(eval REPO := $(shell $(RUN) python -c "from sherlock.config import cfg; print(cfg.demo.model)"))
	$(RUN) hf upload $(REPO) models/legacy/camembert_v6_meta .
	$(RUN) hf upload $(REPO) docs/model_card.md README.md

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

docker-run: ## Lance la démo dans Docker sur http://localhost:8501
	docker run --rm -p 8501:8501 -v sherlock_hf:/root/.cache/huggingface sherlock-demo
