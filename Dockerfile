# Image de la démo Streamlit (CPU). Les poids ne sont pas embarqués (440 Mo) :
#   docker run -p 8501:8501 -v "$(pwd)/models:/app/models" ghcr.io/liliandoublet/sherlock:latest
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
RUN uv pip install ".[data,ml,viz,demo]"

COPY params.yaml ./
COPY app ./app
COPY reports ./reports

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
