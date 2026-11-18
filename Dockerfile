# Pinned official vLLM v0.7.3 base image (CUDA 12.1, Python 3.12, Ubuntu 22.04)
# Digest verified from registry-1.docker.io in DECISIONS.md §1 & §4
FROM vllm/vllm-openai:v0.7.3@sha256:4f4037303e8c7b69439db1077bb849a0823517c0f785b894dc8e96d58ef3a0c2

# Install system dependencies: tini for PID 1 process reaping and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Architecture Invariant 1: Isolated gateway virtual environment
# Gateway dependencies are decoupled from vLLM's system-level packages
RUN python3 -m venv /opt/gateway && \
    /opt/gateway/bin/pip install --no-cache-dir uv && \
    rm -rf /root/.cache /tmp/*

# Environment settings for gateway virtual environment and data paths
ENV VIRTUAL_ENV=/opt/gateway \
    PATH="/opt/gateway/bin:/usr/local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    YLLM_DATA_DIR=/data \
    HF_HOME=/data/models/huggingface \
    YLLM_VLLM_PYTHON=/usr/bin/python3

WORKDIR /app

# Copy entrypoint script, packaging metadata, and source code
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY pyproject.toml README.md ./
COPY src/ /app/src/

# Install gateway package into isolated venv, populate templates, symlink CLI, and configure entrypoint
RUN uv pip install --python /opt/gateway/bin/python --no-cache-dir -e . && \
    mkdir -p /opt/yourllm/templates && \
    cp -r /app/src/yourllm/models/templates/. /opt/yourllm/templates/ && \
    ln -sf /opt/gateway/bin/yourllm-cli /usr/local/bin/yourllm-cli && \
    sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh && \
    rm -rf /root/.cache /tmp/*

# Declare persistent storage volume
VOLUME /data

# Gateway HTTP port
EXPOSE 8000

# Liveness healthcheck (probes gateway liveness only per CLAUDE.md §3 & M4)
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health/live || exit 1

# Process supervisor entrypoint
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]

# Default entrypoint command running YourLLM gateway
CMD ["/opt/gateway/bin/python", "-m", "yourllm.main"]
