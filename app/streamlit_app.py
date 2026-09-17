"""
Démo Streamlit de Sherlock.

Lancer depuis la racine du projet :
    make demo
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sherlock.config import cfg
from sherlock.model.features import meta_prefix
from sherlock.report import render

MODELS_DIR = Path(cfg.paths.models_dir)
REPORTS_DIR = Path(cfg.paths.reports_dir)
TEST_SPLIT = Path(cfg.paths.processed_dir) / "test.parquet"

ACCENT = "#2a78d6"  # barre mise en évidence (prédiction)
NEUTRAL = "#8d8c86"  # autres barres ; ≥ 3:1 sur fond clair et sombre
SENTIMENTS = ["négatif", "neutre", "positif"]

st.set_page_config(page_title="Sherlock", page_icon="🔍", layout="wide")


# ── Chargements (mis en cache) ────────────────────────────────────────────────


def find_models() -> list[Path]:
    """Dossiers de modèles au format save_pretrained ; runs actuels avant l'historique."""
    dirs = {path.parent for path in MODELS_DIR.glob("**/config.json")}
    return sorted(dirs, key=lambda p: ("legacy" in p.parts, str(p)))


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_predictor(model_dir: str):
    from sherlock.model.predict import Predictor

    return Predictor(Path(model_dir))


@st.cache_data
def load_test_split() -> pd.DataFrame | None:
    return pd.read_parquet(TEST_SPLIT) if TEST_SPLIT.exists() else None


@st.cache_data
def load_runs() -> dict[str, dict]:
    return {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(REPORTS_DIR.glob("metrics/*/test_metrics.json"))
    }


def run_for_model(model_dir: Path) -> tuple[str, dict] | None:
    for name, result in load_runs().items():
        if result.get("model_dir") and Path(result["model_dir"]) == model_dir:
            return name, result
    return None


# ── Graphiques ────────────────────────────────────────────────────────────────


def bar_layout(fig: go.Figure, height: int, x_range: list[float]) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        barcornerradius=4,
        showlegend=False,
        xaxis={"range": x_range, "tickformat": ".0%", "gridwidth": 1, "zeroline": False},
        yaxis={"showgrid": False},
    )
    return fig


def probability_chart(scores: dict[str, float], predicted: str, true_party: str | None):
    """Probabilités par parti : la prédiction en bleu, le parti réel marqué dans son libellé."""
    items = sorted(scores.items(), key=lambda kv: kv[1])
    labels = [f"{party}  (parti réel)" if party == true_party else party for party, _ in items]
    fig = go.Figure(
        go.Bar(
            x=[score for _, score in items],
            y=labels,
            orientation="h",
            width=0.6,
            marker_color=[ACCENT if party == predicted else NEUTRAL for party, _ in items],
            text=[
                f"{score:.0%}" if party in (predicted, true_party) else "" for party, score in items
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y} : %{x:.1%}<extra></extra>",
        )
    )
    return bar_layout(fig, height=340, x_range=[0, 1.1])


def f1_by_party_chart(result: dict):
    """F1 par parti pour un run ; seuls le meilleur et le moins bon sont étiquetés."""
    f1 = {label: result["report"][label]["f1-score"] for label in result["labels"]}
    items = sorted(f1.items(), key=lambda kv: kv[1])
    extremes = {items[0][0], items[-1][0]}
    fig = go.Figure(
        go.Bar(
            x=[score for _, score in items],
            y=[party for party, _ in items],
            orientation="h",
            width=0.6,
            marker_color=ACCENT,
            text=[f"{score:.2f}" if party in extremes else "" for party, score in items],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y} : F1 %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_tickformat=".1f")
    return bar_layout(fig, height=320, x_range=[0, 1])


# ── Barre latérale : choix du modèle ──────────────────────────────────────────

st.sidebar.title("🔍 Sherlock")
st.sidebar.caption("Classification du parti politique d'un texte français (8 partis).")

models = find_models()
if not models:
    st.sidebar.error("Aucun modèle dans `models/`.")
    st.error(
        "Aucun modèle trouvé. Lance `make train`, ou place le modèle historique dans "
        "`models/legacy/camembert_v6_meta/`."
    )
    st.stop()

model_dir = st.sidebar.selectbox(
    "Modèle", models, format_func=lambda p: str(p.relative_to(MODELS_DIR))
)
use_meta = st.sidebar.toggle(
    "Sentiment + ironie en entrée",
    value="nometa" not in model_dir.name,
    help="Doit correspondre à la façon dont le modèle a été entraîné.",
)
run = run_for_model(model_dir)
if run:
    name, result = run
    st.sidebar.metric("F1 macro (test)", f"{result['overall']['f1_macro']:.3f}")
    st.sidebar.caption(f"Run MLflow : `{name}`")

predictor = load_predictor(str(model_dir))


# ── Onglets ───────────────────────────────────────────────────────────────────

tab_predict, tab_results = st.tabs(["Prédire", "Résultats"])

with tab_predict:
    test_df = load_test_split()

    def pick_example():
        row = test_df.sample(1).iloc[0]
        st.session_state.texte = row["texte"]
        st.session_state.sentiment = row["sentiment"]
        st.session_state.ironie = bool(row["ironie"])
        st.session_state.exemple = row.to_dict()

    st.session_state.setdefault("texte", "")
    st.session_state.setdefault("sentiment", "neutre")
    st.session_state.setdefault("ironie", False)

    col_btn, col_hint = st.columns([1, 3])
    col_btn.button(
        "🎲 Exemple du test set",
        on_click=pick_example,
        disabled=test_df is None,
        help=None if test_df is not None else "Lance `make data` pour activer les exemples.",
    )
    col_hint.caption(
        "Tire un texte jamais vu à l'entraînement et affiche son vrai parti à côté de la prédiction."
    )

    texte = st.text_area(
        "Texte",
        key="texte",
        height=140,
        placeholder="Collez un tweet ou un extrait de discours politique…",
    )
    col_sent, col_iro = st.columns(2)
    sentiment = col_sent.selectbox("Sentiment", SENTIMENTS, key="sentiment", disabled=not use_meta)
    ironie = col_iro.checkbox("Ironique", key="ironie", disabled=not use_meta)

    exemple = st.session_state.get("exemple")
    true_party = exemple["parti"] if exemple and exemple["texte"] == texte else None

    if texte.strip():
        prediction = predictor.predict([texte], [sentiment], [ironie], use_meta=use_meta)[0]
        parti, confidence = prediction["parti"], prediction["confidence"]

        col_metric, col_chart = st.columns([1, 2])
        with col_metric:
            st.metric("Parti prédit", parti, f"confiance {confidence:.0%}", delta_color="off")
            if true_party:
                if true_party == parti:
                    st.success(f"✓ Correct : parti réel {true_party} ({exemple['media']})")
                else:
                    st.error(f"✗ Erreur : parti réel {true_party} ({exemple['media']})")
        with col_chart:
            st.plotly_chart(
                probability_chart(prediction["all_scores"], parti, true_party),
                width="stretch",
                config={"displayModeBar": False},
            )

        with st.expander("Entrée exacte envoyée au modèle"):
            prefix = meta_prefix(sentiment, ironie) if use_meta else ""
            st.code(prefix + texte, language=None, wrap_lines=True)
    else:
        st.info("Saisis un texte ou tire un exemple du test set.")

with tab_results:
    runs = load_runs()
    st.markdown(render(REPORTS_DIR))

    if runs:
        st.divider()
        name = st.selectbox("Détail d'un run", list(runs))
        result = runs[name]
        overall, by_media = result["overall"], result.get("by_media", {})

        cols = st.columns(4)
        cols[0].metric("F1 macro", f"{overall['f1_macro']:.3f}")
        cols[1].metric("Accuracy", f"{overall['accuracy']:.3f}")
        cols[2].metric("F1 Twitter", f"{by_media.get('Twitter', {}).get('f1_macro', 0):.3f}")
        cols[3].metric("F1 site web", f"{by_media.get('site_web', {}).get('f1_macro', 0):.3f}")

        col_f1, col_cm = st.columns(2)
        with col_f1:
            st.subheader("F1 par parti")
            st.plotly_chart(
                f1_by_party_chart(result), width="stretch", config={"displayModeBar": False}
            )
            table = pd.DataFrame(result["report"]).T.loc[result["labels"]]
            st.dataframe(
                table[["precision", "recall", "f1-score", "support"]].round(3), width="stretch"
            )
        with col_cm:
            st.subheader("Matrice de confusion")
            image = REPORTS_DIR / "metrics" / name / "test_confusion_matrix.png"
            if image.exists():
                st.image(str(image), caption="Lignes : parti réel · colonnes : parti prédit")
