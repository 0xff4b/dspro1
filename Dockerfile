FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY requirements-app.lock .
RUN pip install --require-hashes --only-binary=:all: -r requirements-app.lock
COPY --chown=app:app .streamlit/config.toml .streamlit/config.toml
COPY --chown=app:app src/assets/ src/assets/
COPY --chown=app:app src/app.py src/app.py
COPY --chown=app:app src/plot_style.py src/plot_style.py
COPY --chown=app:app src/notebooks/models/*.joblib src/notebooks/models/
COPY --chown=app:app src/external-sources/output_csv/model.csv src/external-sources/output_csv/model.csv
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"
CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none"]
