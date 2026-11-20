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

# The published MiDaS graph takes one frame per run. Batches above 1 use a copy
# whose batch dimension is symbolic.
COPY scripts/patch_midas.py scripts/patch_midas.py
RUN python scripts/patch_midas.py \
    "$MODEL_DIR/midas_small.onnx" "$MODEL_DIR/midas_small_batched.onnx"

# The container sees every host core, and thread pools sized from that count
# oversubscribe the 2-CPU quota.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# The service runs as a user that owns /app, where the tests write their caches.
# The model files above stay root-owned and readable by everyone.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin rri && chown rri:rri /app

COPY --chown=rri:rri . .

# Set late, so a new commit rebuilds only the layers from here on.
ARG GIT_COMMIT=unknown
ENV RRI_VERSION=$GIT_COMMIT

# Nothing here needs root: the port is unprivileged, a worker's niceness only
# rises, the worker link is an inherited socket pair, and any user can read the
# cgroup and /proc files. A process that writes under the bind-mounted
# /app/results is started with the host user's ids (-u), because this user's id
# need not match the directory's owner.
USER rri

EXPOSE 8080

CMD ["python", "-m", "cloud.main"]
