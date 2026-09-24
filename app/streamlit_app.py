"""
Démo grand public de Sherlock : quel parti se cache derrière ce texte ?

    make demo          # modèle publié sur Hugging Face (téléchargé au premier lancement)
    make demo-local    # modèle local models/legacy/camembert_v6_meta

Le modèle à servir se choisit avec SHERLOCK_MODEL (identifiant Hub ou dossier local) ;
par défaut c'est `demo.model` dans params.yaml.
"""

import json
import os
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from sherlock.config import cfg

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / cfg.paths.reports_dir / "metrics"

ACCENT = "#2a78d6"  # barre mise en évidence
NEUTRAL = "#8d8c86"  # autres barres ; ≥ 3:1 sur fond clair et sombre
HESITATION = 0.40  # sous ce seuil de confiance, on dit que le modèle hésite
SENTIMENTS = ["négatif", "neutre", "positif"]
MAX_WORDS = 350  # ≈ 512 tokens : au-delà, le texte est tronqué
GITHUB = "https://github.com/liliandoublet/sherlock"

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

st.set_page_config(page_title="Sherlock", page_icon="🔍", layout="centered")


# ── Chargements (mis en cache) ────────────────────────────────────────────────


def default_model_source() -> str:
    return os.environ.get("SHERLOCK_MODEL") or cfg.demo.model


@st.cache_resource(
    show_spinner="Chargement du modèle (au premier lancement : ~420 Mo à télécharger)…"
)
def load_predictor(source: str):
    from sherlock.model.predict import Predictor

    return Predictor(source)


@st.cache_data
def load_metrics() -> dict[str, dict]:
    return {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(METRICS_DIR.glob("*/test_metrics.json"))
    }


def pct(value: float) -> str:
    return f"{value * 100:.1f} %".replace(".", ",")


# ── Graphiques ────────────────────────────────────────────────────────────────


def bar_chart(
    labels: list[str],
    values: list[float],
    colors: list[str] | str,
    texts: list[str],
    hover: str,
    height: int,
    x_max: float = 1.0,
) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            width=0.6,
            marker_color=colors,
            text=texts,
            textposition="outside",
            cliponaxis=False,
            hovertemplate=hover,
        )
    )
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        barcornerradius=4,
        showlegend=False,
        xaxis={"range": [0, x_max], "tickformat": ".0%", "gridwidth": 1, "zeroline": False},
        yaxis={"showgrid": False},
    )
    return fig


def probability_chart(scores: dict[str, float], predicted: str) -> go.Figure:
    """Probabilités par parti : la prédiction en bleu, valeur affichée sur les deux premiers."""
    items = sorted(scores.items(), key=lambda kv: kv[1])
    top_two = {party for party, _ in items[-2:]}
    return bar_chart(
        labels=[party for party, _ in items],
        values=[score for _, score in items],
        colors=[ACCENT if party == predicted else NEUTRAL for party, _ in items],
        texts=[f"{score:.0%}" if party in top_two else "" for party, score in items],
        hover="%{y} : %{x:.1%}<extra></extra>",
        height=340,
        x_max=1.1,
    )


def f1_by_party_chart(result: dict) -> go.Figure:
    f1 = {label: result["report"][label]["f1-score"] for label in result["labels"]}
    items = sorted(f1.items(), key=lambda kv: kv[1])
    extremes = {items[0][0], items[-1][0]}
    return bar_chart(
        labels=[party for party, _ in items],
        values=[score for _, score in items],
        colors=ACCENT,
        texts=[f"{score:.2f}" if party in extremes else "" for party, score in items],
        hover="%{y} : F1 %{x:.3f}<extra></extra>",
        height=320,
    )


def comparison_chart(rows: list[tuple[str, float]], highlight: str) -> go.Figure:
    """Comparaison de scores F1 : la ligne mise en avant en bleu, les autres en gris."""
    rows = sorted(rows, key=lambda row: row[1])
    return bar_chart(
        labels=[name for name, _ in rows],
        values=[score for _, score in rows],
        colors=[ACCENT if name == highlight else NEUTRAL for name, _ in rows],
        texts=[pct(score) for _, score in rows],
        hover="%{y} : %{x:.1%}<extra></extra>",
        height=230,
        x_max=0.8,
    )


def show_analysis(texte: str) -> None:
    """Prédit et affiche le résultat ; affiche une aide claire si le modèle est introuvable."""
    source = st.session_state.get("model_source", default_model_source())
    try:
        predictor = load_predictor(source)
    except FileNotFoundError as error:
        st.error(str(error))
        st.info(
            "Vérifie ta connexion Internet (premier lancement), ou utilise un modèle local : "
            "`make demo-local`."
        )
        return

    use_annotations = st.session_state.get("use_annotations", False)
    sentiment = st.session_state.get("sentiment", "neutre") if use_annotations else "neutre"
    irony = st.session_state.get("irony", False) if use_annotations else False
    use_meta = st.session_state.get("use_meta", True)

    with st.spinner("Analyse en cours…"):
        result = predictor.predict([texte], [sentiment], [irony], use_meta=use_meta)[0]

    ranked = sorted(result["all_scores"].items(), key=lambda kv: -kv[1])
    (top, top_score), (second, second_score) = ranked[0], ranked[1]
    n_words = len(texte.split())

    if n_words > MAX_WORDS:
        st.info(f"Texte long : seuls les {MAX_WORDS} premiers mots environ sont analysés.")
    elif n_words < 5:
        st.info("Texte très court : le résultat est peu fiable.")

    hesitates = top_score < HESITATION
    if hesitates:
        st.warning(
            f"Le modèle hésite : **{top}** ({pct(top_score)}) ou **{second}** "
            f"({pct(second_score)}). Ce texte ne contient pas de signal assez net."
        )

    col_result, col_chart = st.columns([1, 2])
    with col_result:
        st.metric(
            "Piste la plus probable" if hesitates else "Parti le plus probable",
            top,
            f"confiance {top_score:.0%}",
            delta_color="off",
        )
        st.caption(f"Deuxième hypothèse : {second} ({pct(second_score)})")
    with col_chart:
        st.plotly_chart(
            probability_chart(result["all_scores"], top),
            width="stretch",
            config={"displayModeBar": False},
        )

    caption = (
        "Estimation statistique : le modèle devine le parti **de l'auteur** d'après le style et "
        "le vocabulaire, pas la position idéologique du texte."
    )
    reference = load_metrics().get("legacy_v6_public_input")
    if reference:
        errors = pct(1 - reference["overall"]["accuracy"])
        caption += f" Il se trompe dans {errors} des cas sur nos textes de test."
    st.caption(caption)


# ── En-tête ───────────────────────────────────────────────────────────────────

st.title("🔍 Sherlock")
st.subheader("Quel parti politique français se cache derrière ce texte ?")
st.caption(
    "Un modèle CamemBERT entraîné sur des tweets et des sites officiels de 8 partis : "
    "EELV, LFI, LR, PCF, PS, Reconquête, Renaissance et RN."
)

tab_try, tab_perf, tab_about = st.tabs(["Essayer", "Performances", "À propos"])


# ── Onglet 1 : essayer ────────────────────────────────────────────────────────

with tab_try:
    st.session_state.setdefault("texte", "")

    def use_example(text: str) -> None:
        st.session_state.texte = text
        st.session_state.lance = True

    st.markdown("**Essayer avec un exemple**")
    st.caption(
        "Phrases rédigées pour la démo. Elles montrent le fonctionnement, pas la précision : "
        "le modèle se trompe aussi, surtout entre partis proches."
    )
    for row in (EXAMPLES[:3], EXAMPLES[3:]):
        for column, (label, text) in zip(st.columns(3), row, strict=True):
            column.button(
                label,
                key=f"ex_{label}",
                on_click=use_example,
                args=(text,),
                width="stretch",
            )

    with st.form("analyse"):
        st.text_area(
            "Ou colle ton propre texte (tweet, communiqué, extrait de discours…)",
            key="texte",
            height=140,
            max_chars=3000,
        )
        submitted = st.form_submit_button("Analyser", type="primary")
    if submitted:
        st.session_state.lance = True

    texte = st.session_state.texte.strip()
    if st.session_state.get("lance") and not texte:
        st.warning("Écris ou colle d'abord un texte.")
    elif st.session_state.get("lance"):
        show_analysis(texte)

    with st.expander("Options avancées"):
        st.text_input(
            "Modèle (dossier local ou identifiant Hugging Face)",
            value=default_model_source(),
            key="model_source",
        )
        st.checkbox(
            "Le modèle attend un préfixe sentiment/ironie (cas des modèles entraînés avec)",
            value=True,
            key="use_meta",
        )
        st.toggle(
            "Renseigner moi-même le sentiment et l'ironie",
            value=False,
            key="use_annotations",
            help="Par défaut : neutre et non ironique, les annotations n'étant pas connues d'avance.",
        )
        st.selectbox("Sentiment", SENTIMENTS, index=1, key="sentiment")
        st.checkbox("Ironique", key="irony")


# ── Onglet 2 : performances ───────────────────────────────────────────────────

with tab_perf:
    runs = load_metrics()
    demo_run = runs.get("legacy_v6_public_input")
    if demo_run is None:
        st.info("Métriques absentes : lance `make all` pour les générer.")
    else:
        n_classes = len(demo_run["labels"])
        rows = [
            (f"Hasard (1 chance sur {n_classes})", 1 / n_classes),
            ("Baseline TF-IDF + régression", runs["tfidf_logreg"]["overall"]["f1_macro"]),
            ("Sherlock (conditions de la démo)", demo_run["overall"]["f1_macro"]),
        ]
        if "legacy_v6_meta" in runs:
            rows.append(
                (
                    "Sherlock + annotations sentiment/ironie",
                    runs["legacy_v6_meta"]["overall"]["f1_macro"],
                )
            )

        n_tests = f"{demo_run['n']:,}".replace(",", " ")
        st.markdown(
            f"**Score F1 macro** sur {n_tests} textes que le modèle n'a jamais vus : "
            f"1 = parfait, {pct(1 / n_classes)} = réponse au hasard."
        )
        st.plotly_chart(
            comparison_chart(rows, "Sherlock (conditions de la démo)"),
            width="stretch",
            config={"displayModeBar": False},
        )

        st.markdown("**Score par parti**")
        st.plotly_chart(
            f1_by_party_chart(demo_run), width="stretch", config={"displayModeBar": False}
        )

        by_media = demo_run.get("by_media", {})
        if by_media:
            columns = st.columns(len(by_media))
            names = {"Twitter": "Sur les tweets", "site_web": "Sur les sites officiels"}
            for column, (media, values) in zip(columns, by_media.items(), strict=True):
                column.metric(names.get(media, media), pct(values["f1_macro"]))

        image = METRICS_DIR / "legacy_v6_public_input" / "test_confusion_matrix.png"
        if image.exists():
            st.markdown("**Qui est confondu avec qui**")
            st.image(
                str(image),
                caption="Chaque ligne est le vrai parti, chaque colonne le parti prédit. "
                "Plus la diagonale est foncée, mieux le parti est reconnu.",
            )


# ── Onglet 3 : à propos ───────────────────────────────────────────────────────

with tab_about:
    st.markdown(
        f"""
**Comment ça marche.** Sherlock est un modèle de langue français (CamemBERT) réentraîné pour
classer un texte parmi 8 partis. Il a appris sur 14 696 textes : des tweets de 24 comptes de
personnalités politiques et des extraits de sites officiels, de 2019 à 2025.

**Ce que « parti » veut dire ici.** Le modèle prédit le parti *de l'auteur du texte*, d'après son
style et son vocabulaire. Il ne mesure ni la position idéologique d'un texte, ni sa véracité.

**Limites.**
- Il se trompe régulièrement, surtout entre partis proches (LR/RN, PS/PCF, RN/Reconquête).
- Un texte factuel ou sans vocabulaire politique ne contient aucun signal : le modèle hésite.
- Les opinions politiques sont une donnée sensible : n'utilise pas cet outil pour profiler une
  personne réelle.

**Vie privée.** Le modèle s'exécute sur ta machine. Le texte que tu saisis n'est ni enregistré ni
envoyé à un service tiers ; seul le téléchargement initial du modèle utilise Internet.

**Code, données et méthode** : [{GITHUB}]({GITHUB})
"""
    )
