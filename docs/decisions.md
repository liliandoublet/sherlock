# Decisions log

Journal des choix techniques du projet Sherlock.
Chaque décision explique le contexte, les options considérées, et le choix final.

---

## 001 - CamemBERT-base (révisée)

**Contexte** : choix du modèle de base pour le fine-tuning.

**Options considérées** :
- CamemBERT-base (2019) : architecture RoBERTa, 110M paramètres
- CamemBERTa-v2 (2024) : architecture DeBERTa-v3, corpus plus large et plus récent
- XLM-RoBERTa-large : multilingue, plus généraliste

**Décision initiale** : CamemBERTa-v2 (jamais entraîné).

**Décision révisée** : CamemBERT-base (`almanach/camembert-base`).

**Raisons** : les modèles historiques V3 à V8 sont des CamemBERT-base (vérifié dans leurs
`config.json`), entraînés sur les mêmes splits : F1 macro de 0,557 à 0,646, dont 0,629 pour V6. Garder le même modèle et les mêmes hyperparamètres
que V6 rend le nouveau pipeline directement comparable à l'ancien : un écart de score
s'explique alors par le code, pas par le changement de modèle. CamemBERTa-v2 reste la
prochaine expérience naturelle ; il suffit de changer `model.name` dans `params.yaml`.

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

**Note** : ce seuil concerne le pipeline de reconstruction (`sherlock dataset balance`).
Les splits historiques utilisés pour les résultats comptent environ 1 837 textes par parti
(voir 007).

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

---

## 007 - Conserver les splits historiques tels quels

**Contexte** : le nouveau pipeline doit être évalué, alors que le corpus du projet de fin
d'études est déjà découpé (11 756 / 1 470 / 1 470).

**Options considérées** :
- Re-découper le corpus avec `sherlock dataset split` (seed 42)
- Réutiliser les fichiers `train.csv` / `val.csv` / `test.csv` d'origine

**Décision** : réutiliser les splits d'origine (`sherlock dataset import-legacy`).

**Raisons** : c'est la seule façon de comparer les nouveaux runs aux modèles historiques
sur exactement les mêmes textes de test. La ré-évaluation du modèle V6 (0,627 contre 0,629
exporté à l'époque) valide la chaîne complète : données, préfixes, évaluation.

**Défauts assumés** : 8 textes identiques entre train et test, 7 entre train et val, 1 entre
val et test. C'est négligeable (0,5 % du test set), et l'import les signale au lieu de les
supprimer. Corriger 5 coquilles de sentiment (`positre`) ne touche pas le test set.

---

## 008 - Sentiment et ironie injectés comme préfixes du texte

**Contexte** : chaque texte est annoté (sentiment, ironie) par Gemini. Faut-il donner ces
informations au classifieur ?

**Options considérées** :
- Préfixes textuels : `[NOIRONY] [SENT_négatif] texte…` (modèles historiques V3, V4, V6)
- Features numériques concaténées au vecteur [CLS] (modèle V8, architecture sur mesure)
- Texte seul

**Décision** : préfixes textuels par défaut (`model.use_meta: true`), texte seul disponible
via `sherlock train --no-meta`.

**Raisons** : les préfixes gardent le format standard `AutoModelForSequenceClassification`,
donc le même code sert à entraîner, évaluer et servir les modèles historiques et nouveaux.
V8 (features numériques, 0,646) demande une architecture et un chargement spécifiques.

**Ce qu'on ne sait pas encore** : l'apport réel de ces métadonnées. Le dossier historique
« V4 Wo Sent and ironie » semblait fournir l'ablation (0,627 contre 0,629 pour V6), mais le
modèle V4 réagit aux préfixes exactement comme V6 : 0,627 avec, 0,615 sans. Son score
d'origine a donc été mesuré **avec** les préfixes, et ce n'est pas une ablation. La
comparaison fiable est `make train` contre `make train-ablation`, sur les mêmes splits et
avec les mêmes hyperparamètres.

**Point d'attention** : en production, sentiment et ironie ne sont pas connus à l'avance.
Il faudrait les prédire (module `annotate`) ou utiliser le modèle texte seul. La démo laisse
l'utilisateur les choisir.
