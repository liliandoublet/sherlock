# Decisions log

Journal des choix techniques du projet Sherlock.
Chaque décision explique le contexte, les options considérées, et le choix final.

---

## 001 - CamemBERTa-v2 plutôt que CamemBERT-base

**Contexte** : choix du modèle de base pour le fine-tuning.

**Options considérées** :
- CamemBERT-base (2019) : architecture BERT, 110M paramètres
- CamemBERTa-v2 (2023) : architecture DeBERTa, corpus plus large
- XLM-RoBERTa-large : multilingue, plus généraliste

**Décision** : CamemBERTa-v2.

**Raisons** : architecture DeBERTa supérieure à BERT sur les tâches de
classification fine-grained. Corpus d'entraînement plus récent et plus large
que CamemBERT-base. Préféré à XLM-RoBERTa car monolingue français,
plus adapté à notre corpus politique.

---

## 002 - Gemini pour le nettoyage web plutôt que regex seules

**Contexte** : nettoyage des communiqués de partis politiques scrapés.

**Options considérées** :
- Regex uniquement : rapide, gratuit, déterministe
- Gemini API : lent, coût, mais compréhension contextuelle

**Décision** : Gemini pour le web, regex pour Twitter.

**Raisons** : les communiqués contiennent des structures complexes
(listes de fonctions politiques, signatures, mentions promotionnelles)
impossibles à détecter avec des regex sans faux positifs massifs.
Twitter en revanche a des patterns simples et uniformes (URLs, mentions,
hashtags) parfaitement gérés par des regex.

---

## 003 - Équilibrage par under-sampling plutôt que class weights

**Contexte** : déséquilibre entre partis dans le corpus.

**Options considérées** :
- Class weights dans la loss : garde toutes les données
- Under-sampling : perd des données mais simplifie
- Over-sampling (SMOTE) : risque de sur-apprentissage sur textes synthétiques

**Décision** : under-sampling à 1392 exemples par parti.

**Raisons** : class weights compliquent l'interprétation des métriques.
SMOTE sur du texte produit des exemples peu naturels. 1392 est le nombre
d'exemples du parti le moins représenté après nettoyage, ce qui garantit
un corpus 100% réel sans duplication.

---

## 004 - Stratification sur `parti` seul au lieu de `parti × media`

**Contexte** : choix de la variable de stratification pour le split train/val/test.

**Options considérées** :
- Stratifier sur `parti` seul
- Stratifier sur `parti × media` (tweet vs communiqué)

**Décision** : `parti` seul.

**Raisons** : stratifier sur `parti × media` introduit un leak implicite.
Le modèle apprend à distinguer le style d'écriture d'un tweet vs d'un
communiqué, corrélé au parti, plutôt que l'idéologie politique elle-même.
Un modèle évalué uniquement sur des tweets aurait des perfs artificiellement
bonnes si le train set était dominé par des tweets du même parti.

---

## 005 - Parquet plutôt que CSV pour les splits finaux

**Contexte** : format de stockage des datasets train/val/test.

**Options considérées** :
- CSV pipe-delimited : universel, lisible par tous les outils
- Parquet : binaire, moins portable mais plus performant

**Décision** : Parquet.

**Raisons** : 10× plus rapide à lire que CSV pour pandas/PyTorch.
Typage fort des colonnes (pas de conversion implicite). Compression native
réduit l'espace disque de ~60%. Pour des itérations rapides de fine-tuning,
le gain de temps de chargement est significatif.

---

## 006 - uv plutôt que pip + virtualenv

**Contexte** : gestion des dépendances Python.

**Options considérées** :
- pip + requirements.txt : standard, mais lent et pas de lock file strict
- poetry : lock file, mais lent à résoudre
- uv : lock file, très rapide, remplace aussi pyenv

**Décision** : uv.

**Raisons** : 10-100× plus rapide que pip pour l'installation.
`uv.lock` garantit des builds reproductibles. Remplace pip, virtualenv
et pyenv en un seul outil. Standard en 2026 pour les nouveaux projets Python.