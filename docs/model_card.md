---
language: fr
license: mit
base_model: almanach/camembert-base
pipeline_tag: text-classification
tags:
  - camembert
  - french
  - political-science
  - text-classification
---

# Sherlock : CamemBERT pour deviner le parti d'un texte politique français

Classe un texte français parmi 8 partis : **EELV, LFI, LR, PCF, PS, Reconquête, Renaissance, RN**.
Modèle `almanach/camembert-base` fine-tuné (modèle « V6 » du projet
[liliandoublet/sherlock](https://github.com/liliandoublet/sherlock)).

## Utilisation

Le modèle a été entraîné avec un préfixe sentiment/ironie. Sans annotation, utiliser le préfixe neutre :

```python
from transformers import pipeline

clf = pipeline("text-classification", model="liliandoublet/sherlock-camembert-v6", top_k=None)
clf("[NOIRONY] [SENT_neutre] Il faut baisser les impôts de production et soutenir les entreprises.")
```

Préfixes possibles : `[NOIRONY]` ou `[IRONY]`, puis `[SENT_négatif]`, `[SENT_neutre]` ou `[SENT_positif]`.

## Performances

Test set de 1 470 textes jamais vus (8 classes équilibrées, hasard = 12,5 %).

| Conditions | F1 macro |
|---|---|
| Préfixe neutre (usage réel) | 0,626 |
| Sentiment et ironie annotés | 0,627 |
| Baseline TF-IDF + régression logistique | 0,570 |

Tweets : ≈ 0,59. Extraits de sites officiels : ≈ 0,73. Méthode et reproduction :
[docs/reproduce.md](https://github.com/liliandoublet/sherlock/blob/main/docs/reproduce.md).

## Limites et usage responsable

- Le modèle prédit le parti **de l'auteur**, d'après le style et le vocabulaire : il ne mesure pas une
  position idéologique. Il se trompe dans près de 4 cas sur 10, surtout entre partis proches.
- Entraîné sur 14 696 textes (tweets de 24 comptes de personnalités politiques, extraits de sites
  officiels, 2019-2025) : il généralise mal à d'autres registres.
- Les opinions politiques sont une donnée sensible : ne pas utiliser ce modèle pour profiler une personne.
- Le corpus d'entraînement n'est pas redistribué.
