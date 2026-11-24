---
title: Sherlock
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Quel parti politique se cache derrière ce texte ?
models:
  - __MODEL__
---

# Sherlock

Démo de [Sherlock](https://github.com/liliandoublet/sherlock) : un CamemBERT fine-tuné qui estime
lequel de 8 partis politiques français a écrit un texte (EELV, LFI, LR, PCF, PS, Reconquête,
Renaissance, RN).

Le modèle prédit le parti **de l'auteur**, d'après le style et le vocabulaire : il ne mesure pas la
position idéologique d'un texte, et se trompe dans près de 4 cas sur 10, surtout entre partis
proches. Les opinions politiques sont une donnée sensible : ne pas l'utiliser pour profiler une
personne réelle.

Ce Space est publié par `make publish-space` depuis le dépôt GitHub : ne pas le modifier à la main.
