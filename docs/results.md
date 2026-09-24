# Résultats détaillés

Page **générée** par `make report` à partir de `reports/` : ne pas modifier à la main.

**Runs du pipeline actuel** (test set, 1 470 textes, suivis dans MLflow) :

| Modèle | Run | Accuracy | F1 macro | F1 Twitter | F1 site web |
|---|---|---|---|---|---|
| V6 dans les conditions de la démo (sans annotations) | `legacy_v6_public_input` | 0,624 | **0,626** | 0,591 | 0,733 |
| CamemBERT fine-tuné, texte + sentiment/ironie | `camembert_meta` | *non lancé (`make train`)* | | | |
| CamemBERT fine-tuné, texte seul (ablation) | `camembert_nometa` | *non lancé (`make train-ablation`)* | | | |
| V6 avec sentiment/ironie annotés (référence du CV) | `legacy_v6_meta` | 0,626 | **0,627** | 0,594 | 0,730 |
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
