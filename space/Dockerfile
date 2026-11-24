# Image de la démo en ligne, pour un Space Hugging Face (SDK Docker). Publiée par `make publish-space`.
# Différences avec le Dockerfile racine :
#   - le modèle (demo.model dans params.yaml) est téléchargé au build : démarrage rapide, et un
#     dépôt privé reste possible avec un secret HF_TOKEN dans les réglages du Space ;
#   - l'appli tourne avec l'utilisateur 1000 imposé par les Spaces, HOME pointe vers /tmp.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

WORKDIR /app

# torch CPU en premier : le reste des dépendances le réutilise au lieu de tirer CUDA
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md params.yaml ./
COPY src ./src
RUN uv pip install ".[demo]"

RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=false \
    if [ -s /run/secrets/HF_TOKEN ]; then export HF_TOKEN="$(cat /run/secrets/HF_TOKEN)"; fi; \
    python -c "from huggingface_hub import snapshot_download; from sherlock.config import cfg; \
snapshot_download(cfg.demo.model, local_dir='/app/model')"

COPY .streamlit ./.streamlit
COPY app ./app
COPY reports ./reports

ENV SHERLOCK_MODEL=/app/model \
    SHERLOCK_HOSTED=1 \
    HOME=/tmp

USER 1000
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
