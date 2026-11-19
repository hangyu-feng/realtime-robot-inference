FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/opt/models

WORKDIR /app

# Dependencies first, so a code change does not reinstall them.
COPY requirements.txt .
RUN pip install --require-hashes -r requirements.txt

# Both models are fetched at build time and verified by SHA-256.
COPY scripts/fetch_models.py scripts/fetch_models.py
RUN python scripts/fetch_models.py "$MODEL_DIR"

COPY . .

# ASSUMPTION: the container runs as root. Results are written to a bind mount,
# and root avoids a user-id mismatch with whoever owns that directory on the host.

EXPOSE 8080

# Placeholder until the gateway exists: print the limits the container runs under.
CMD ["python", "-m", "common.cgroup"]
