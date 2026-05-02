# ── Base image ──────────────────────────────────────────────────────────────
FROM python:3.9-slim

# ── Metadata ─────────────────────────────────────────────────────────────────
LABEL maintainer="mlops-assessment"
LABEL description="MLOps batch signal pipeline — MetaStackerBandit T0"

# ── System deps (minimal) ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (layer-cached before copying source) ─────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ───────────────────────────────────────────────────────
COPY run.py      .
COPY config.yaml .
COPY data.csv    .

# ── Default command — no hard-coded paths, all passed via CLI flags ──────────
CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]
