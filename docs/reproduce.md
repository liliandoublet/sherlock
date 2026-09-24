# Tout reproduire

## Prérequis

- [uv](https://docs.astral.sh/uv/) et Python 3.12
- Un GPU NVIDIA pour le fine-tuning (testé sur RTX 3060 Ti, 8 Go). L'évaluation et la démo tournent aussi sur CPU.
- Deux ensembles de fichiers non versionnés (trop lourds ou non redistribuables) :

  | Fichier | Emplacement attendu |
  |---|---|
  | Splits historiques du corpus | `data/raw/legacy/{train,val,test}.csv` (séparateur `\|`) |
  | Modèle V6 (pour `eval-legacy` et `demo-local`) | `models/legacy/camembert_v6_meta/` |

## Commandes

```bash
make install          # uv sync --all-extras
make all              # import des données, évaluation de V6, baseline, tableaux (quelques minutes)
make train            # fine-tuning CamemBERT, texte + sentiment/ironie (~10 min sur RTX 3060 Ti)
make train-ablation   # fine-tuning CamemBERT, texte seul
make report           # met à jour le tableau du README et docs/results.md
make mlflow           # http://localhost:5000
make demo             # http://localhost:8501
make help             # toutes les cibles
```

Équivalent sans `make` :

```bash
uv sync --all-extras
uv run sherlock dataset import-legacy data/raw/legacy
uv run sherlock evaluate models/legacy/camembert_v6_meta --run-name legacy_v6_meta
uv run sherlock evaluate models/legacy/camembert_v6_meta --neutral-meta --run-name legacy_v6_public_input
uv run sherlock baselines
uv run sherlock train --run-name camembert_meta --output-dir models/camembert_party
uv run sherlock train --no-meta --run-name camembert_nometa --output-dir models/camembert_nometa
uv run sherlock report
uv run sherlock predict "Il faut sortir du nucléaire" --model-dir models/camembert_party
```

## Suivi des expériences (MLflow)

`train`, `evaluate` et `baselines` créent chacun un run dans l'expérience
`sherlock-party-classification` (base `sqlite:///mlflow.db`, réglée dans `params.yaml`).

| Enregistré | Détail |
|---|---|
| Paramètres | modèle, epochs, batch, learning rate, seed, fp16, `use_meta`, tailles des splits, device |
| Tags | commit git, étape (`training` / `evaluation` / `baseline`), dossier du modèle |
| Métriques par epoch | `train_loss`, `val_accuracy`, `val_f1_macro`, `val_f1_weighted`, durée |
| Métriques finales | `test_*` globales et par média (`test_Twitter_f1_macro`, `test_site_web_f1_macro`…) |
| Artefacts | `params.yaml`, rapport de classification JSON, matrice de confusion PNG |

Les métriques de test sont aussi écrites dans `reports/metrics/<run>/` (versionné), lisibles sans MLflow.

## Modèle

- `almanach/camembert-base` + tête de classification linéaire (`AutoModelForSequenceClassification`)
- Entrée `[NOIRONY] [SENT_négatif] texte…`, tronquée à 512 tokens (`--no-meta` : texte seul)
- AdamW, lr 2e-5, batch 16, 4 epochs, fp16, seed 92 ; meilleur epoch choisi sur la F1 macro de validation
- Hyperparamètres dans [`params.yaml`](../params.yaml), identiques à ceux du modèle historique V6

## Si WSL plante pendant l'entraînement

WSL2 n'alloue par défaut qu'une partie de la RAM de Windows, et l'entraînement, VS Code Server et le
navigateur se la partagent. Dans `%UserProfile%\.wslconfig` (côté Windows), puis `wsl --shutdown`
(valeurs à adapter à la RAM de la machine, par exemple environ deux tiers) :

```ini
[wsl2]
memory=10GB
swap=8GB
```
