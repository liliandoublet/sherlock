# Données

| | |
|---|---|
| Volume | 14 696 textes : 11 756 train / 1 470 val / 1 470 test |
| Classes | 8 partis, équilibrés (≈ 1 837 textes par parti) |
| Sources | 11 136 tweets de 24 comptes de personnalités politiques, 3 560 extraits de sites officiels des partis |
| Période | septembre 2019 → mai 2025 |
| Annotations | sentiment (`positif` / `neutre` / `négatif`) et ironie (booléen), par Gemini 2.0 Flash à température 0 |
| Répartition | 50,8 % négatif, 27,6 % positif, 21,6 % neutre ; 13,2 % ironique |

Le corpus n'est **pas redistribué** dans ce dépôt (tweets et contenus tiers). Les exemples de la démo
sont écrits à la main.

## Colonnes

`texte`, `compte`, `parti`, `date`, `source`, `media` (`Twitter` ou `site_web`), `ironie`, `sentiment`.

## Limites connues

- 8 textes identiques entre train et test, 7 entre train et val, 1 entre val et test. Les splits sont
  conservés tels quels pour rester comparables aux résultats historiques ; `import-legacy` les signale.
- Sentiment et ironie viennent d'un LLM, sans validation humaine systématique.
- 5 coquilles `positre` corrigées en `positif` à l'import (aucune dans le test set).
- Le label est le parti de l'auteur, pas le contenu idéologique du texte : un tweet purement factuel
  (« je serai l'invité politique du 8h30 ») ne porte aucun signal partisan.
- Le corpus couvre 24 comptes et des sites officiels : le modèle généralise mal à d'autres registres
  (presse, forums, conversation).

## Reconstruire un corpus

Les modules `collect`, `clean` et `annotate` (`src/sherlock/`) servent à repartir de zéro ; ils ne sont pas
utilisés pour les résultats publiés, qui reposent sur les splits historiques.
