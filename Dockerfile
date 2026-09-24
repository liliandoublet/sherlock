# Image de la démo Streamlit (CPU). Les poids (~420 Mo) ne sont pas embarqués : ils sont téléchargés
# depuis Hugging Face au premier lancement, puis gardés dans le volume `sherlock_hf`.
#   docker run -p 8501:8501 -v sherlock_hf:/root/.cache/huggingface ghcr.io/liliandoublet/sherlock:latest
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

WORKDIR /app

# torch CPU en premier : le reste des dépendances le réutilise au lieu de tirer CUDA
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install ".[demo]"

COPY params.yaml ./
COPY .streamlit ./.streamlit
COPY app ./app
COPY reports ./reports

# Modèle servi : identifiant Hugging Face ou dossier monté (-e SHERLOCK_MODEL=/models/mon_modele)
ENV SHERLOCK_MODEL=""

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
