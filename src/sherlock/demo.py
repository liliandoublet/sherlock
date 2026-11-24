"""
demo.py
-------
Réglages communs à la démo Streamlit (app/) et au site vitrine (site/) : mêmes exemples, mêmes
seuils, pour que le site décrive exactement ce que fait la démo.
"""

HESITATION = 0.40  # sous ce seuil de confiance, on dit que le modèle hésite
MIN_WORDS = 5  # en dessous, le résultat est peu fiable
MAX_WORDS = 350  # ≈ 512 tokens : au-delà, le texte est tronqué

# Exemples rédigés pour la démo (aucun texte du corpus n'est redistribué).
EXAMPLES = [
    (
        "Retraite à 60 ans",
        "Nous voulons la retraite à 60 ans, le blocage des prix de l'énergie et une VIe République "
        "pour redonner le pouvoir au peuple.",
    ),
    (
        "Nationalisations",
        "Nationalisons les grands services publics et augmentons les salaires : le travail doit "
        "mieux payer que le capital.",
    ),
    (
        "Europe solidaire",
        "La gauche de gouvernement doit protéger les services publics tout en construisant une "
        "Europe plus solidaire et une transition juste.",
    ),
    (
        "Identité et sécurité",
        "Il faut défendre l'identité française, exiger l'assimilation et rétablir la sécurité "
        "dans nos quartiers.",
    ),
    (
        "Immigration (cas limite)",
        "Il faut maîtriser l'immigration, défendre nos frontières et instaurer la priorité "
        "nationale pour les Français.",
    ),
    (
        "Sans signal politique",
        "Merci à toutes et à tous pour votre présence ce soir, belle soirée !",
    ),
]
