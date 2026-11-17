# YourLLM

**Bring Your Own Compute. Deploy Your Own Model. Get Your Own API.**

YourLLM is a single Docker image that turns any NVIDIA GPU machine (cloud instance, homelab server, or local workstation) into a secure, OpenAI-compatible inference server tailored specifically for AI coding agents such as **Cline**, **Roo Code**, and **Continue**.

---

## 1. Key Differentiators

- **Zero-Serialization Streaming:** True byte-level SSE passthrough (`aiter_raw`) without JSON chunk deserialization and re-serialization. Eliminates latency jitter and delivers tool calls and indentation verbatim to coding agents.
- **Client Disconnect Upstream Cancellation:** Detects when an agent cancels generation (e.g. user interruption) and commands vLLM to abort generation immediately, freeing GPU KV cache memory.
- **Curated Model Profiles:** Built-in profiles pairing open-weight coding models with the exact tool-call parsers and bundled chat templates required to avoid parser mismatches.
- **Dependency Isolation:** Gateway dependencies run in an isolated virtual environment (`/opt/gateway`) decoupled from vLLM's system packages.
- **Supervised Process Lifecycle:** Supervises vLLM v0.7.3 in an isolated process group with signal escalation (SIGTERM $\rightarrow$ 15s grace $\rightarrow$ SIGKILL) and bounded automatic recovery.
- **Hardware Validation & VRAM Heuristic:** Pre-validates NVIDIA GPU compute capability ($\ge 7.0$), driver minor-version compatibility, and reports advisory memory estimation ranges before startup.
- **Safe-by-Default Key Management:** Plaintext API keys are never stored; only SHA-256 hashes are persisted. Managed via a local container CLI (`yourllm-cli`).

---

## 2. Hardware & Software Requirements

| Component | Minimum Requirement | Recommended |
|---|---|---|
| **GPU Architecture** | NVIDIA GPU with Compute Capability $\ge 7.0$ (RTX 20/30/40-series, GTX 16-series, T4, V100) | Ampere (RTX 3050/3060/3090), Ada (RTX 4060/4090), Hopper |
| **VRAM** | $\ge 6$ GB (for 1.5B models like Qwen 2.5 Coder 1.5B on RTX 3050 laptops)<br>$\ge 16$ GB (for 7B/8B models) | $\ge 24$ GB (RTX 3090/4090) or cloud GPUs for 32B models |
| **Host Driver** | Linux: $\ge 525.60.13$<br>Windows WSL2: $\ge 527.41$ | Driver 535.x, 550.x, or 570.x+ |
| **Container Engine**| Docker Engine 20.10+ with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) or Docker Desktop (WSL2) | Docker Desktop 4+ / Docker 24+ |
| **Shared Memory** | `--shm-size=4g` (1.5B models) / `--shm-size=8g` (7B/8B models) | `--shm-size=16g` |
| **Storage** | $\ge 15$ GB free on `/data` for 1.5B weights<br>$\ge 30$ GB free for 8B weights | NVMe SSD storage |

---

## 3. Quick Start (Compose-Free Docker)

### Step 1: Configure Environment
Copy `.env.example` to `.env` and configure your model:

```bash
cp .env.example .env
```

**Example for 6 GB Laptop GPUs (RTX 3050/4050) — Qwen 2.5 Coder 1.5B:**
```ini
YLLM_MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
YLLM_SERVED_MODEL_NAME=qwen-coder-1.5b
YLLM_GPU_MEMORY_UTILIZATION=0.90
YLLM_MAX_MODEL_LEN=8192
```

**Example for 16 GB+ GPUs (RTX 3090/4090, Cloud Instances) — Llama 3.1 8B:**
```ini
YLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
YLLM_SERVED_MODEL_NAME=llama-3.1-8b
HF_TOKEN=hf_your_huggingface_access_token_if_gated
YLLM_GPU_MEMORY_UTILIZATION=0.90
YLLM_MAX_MODEL_LEN=32768
```

### Step 2: Get the Docker Image

You can pull the pre-built image directly from Docker Hub or build it locally:

**Option A: Pull pre-built image from Docker Hub (Recommended)**
```bash
docker pull sanjaydocker832/yourllm:latest
```

**Option B: Build from source locally**
```bash
docker build -t sanjaydocker832/yourllm:latest .
```

### Step 3: Run the Container

```bash
docker run -d \
  --name yourllm \
  --gpus all \
  -p 127.0.0.1:8000:8000 \
  --shm-size=8g \
  -v yourllm-data:/data \
  --env-file .env \
  --restart unless-stopped \
  sanjaydocker832/yourllm:latest
```

> [!IMPORTANT]
> **Understanding Port Binding (`-p 127.0.0.1:8000:8000`):**
> Binding port 8000 to `127.0.0.1` restricts access **exclusively to the local host loopback**.
> If your coding agent runs on a separate laptop connecting to a remote GPU server, see [Remote Deployment Architecture](#4-remote-deployment-architecture) below. Never bind directly to `0.0.0.0:8000` without TLS encryption.

### Step 4: Retrieve Bootstrap API Key
On the initial run, YourLLM prints the operational banner containing your first-boot API key:

```bash
docker logs yourllm
```

```text
==============================================================================
  YourLLM Inference Server — READY
==============================================================================
  State:        READY
  Model:        meta-llama/Llama-3.1-8B-Instruct
  Served Name:  llama-3.1-8b
  Hardware:     NVIDIA GeForce RTX 4090 (24.0 GB VRAM, Driver: 550.54.14)
  Base URL:     http://127.0.0.1:8000/v1
  Health URL:   http://127.0.0.1:8000/health/ready
------------------------------------------------------------------------------
  Bootstrap API Key (Created on first boot):
    yllm_a1b2c3d4e5_0123456789abcdefghijklmnopqrstuv

  Configure your coding agent (Cline / Roo Code / Continue) with:
    Base URL: http://127.0.0.1:8000/v1
    API Key:  yllm_a1b2c3d4e5_0123456789abcdefghijklmnopqrstuv
    Model ID: llama-3.1-8b
==============================================================================
```

---

## 4. Remote Deployment Architecture

When your GPU runs on a remote cloud server (RunPod, Lambda, Hetzner, AWS) and your coding agent runs locally on your workstation, use one of the following secure connection patterns:

```mermaid
flowchart LR
    subgraph ClientMachine ["Developer Laptop"]
        Agent["Coding Agent (Cline / Roo Code)"]
        LocalTunnel["Local Port 8000"]
    end

    subgraph SecureTransit ["Encrypted Transit"]
        direction TB
        SSH["Option A: SSH Port Forwarding"]
        VPN["Option B: Tailscale Mesh VPN"]
        TLS["Option C: Reverse Proxy (HTTPS)"]
    end

    subgraph GPUServer ["Remote GPU Server"]
        YourLLM["YourLLM Gateway (127.0.0.1:8000)"]
        vLLM["Internal vLLM (127.0.0.1:8001)"]
        GPU["NVIDIA GPU"]
    end

    Agent --> LocalTunnel
    LocalTunnel -.-> SSH & VPN & TLS
    SSH & VPN & TLS --> YourLLM
    YourLLM --> vLLM --> GPU
```

1. **Option A: SSH Port Forwarding (Recommended for Single Users):**
   ```bash
   ssh -N -L 8000:127.0.0.1:8000 user@gpu-server-ip
   ```
   Configure your coding agent with: `http://127.0.0.1:8000/v1`.
2. **Option B: Private Mesh VPN (Tailscale / WireGuard):**
   Bind the container to the host's private VPN IP: `-p <tailscale-ip>:8000:8000`.
3. **Option C: Reverse Proxy with TLS (Caddy / Nginx):**
   Terminate HTTPS using Caddy or Nginx with valid Let's Encrypt certificates and `proxy_buffering off;`.

---

## 5. API Usage Examples (`curl`)

### 5.1 Non-Streaming Chat Completion
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer yllm_bootstrap_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b",
    "messages": [
      {"role": "system", "content": "You are a concise programming assistant."},
      {"role": "user", "content": "Reverse a string in Python in one line."}
    ],
    "temperature": 0.1
  }'
```

### 5.2 Streaming Chat Completion (`curl -N`)
```bash
curl -N -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer yllm_bootstrap_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b",
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "stream": true
  }'
```

### 5.3 Tool Calling Request (Function Calling)
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer yllm_bootstrap_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b",
    "messages": [{"role": "user", "content": "Read file src/main.py"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "read_file",
          "description": "Reads a local file",
          "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
          }
        }
      }
    ]
  }'
```

---

## 6. Key Management (`yourllm-cli`)

Manage API keys inside the running container using `docker exec`:

```bash
# Create a new named API key
docker exec -it yourllm yourllm-cli key create --name "laptop-cline"

# List active and revoked keys
docker exec -it yourllm yourllm-cli key list

# Rotate an existing key (creates new secret, revokes old)
docker exec -it yourllm yourllm-cli key rotate <key_id>

# Revoke an existing key immediately
docker exec -it yourllm yourllm-cli key revoke <key_id>
```

---

## 7. Curated Model Profiles

YourLLM includes versioned model profiles pairing open-weight models with the correct tool-call parsers and bundled chat templates in `src/yourllm/models/profiles/`:

| Profile Name | Model Repository | Quantization | Tool Parser | Recommended Context | Min VRAM | Supported Hardware |
|---|---|---|---|---|---|---|
| `qwen2.5-coder-1.5b-instruct` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | None (BF16) | `hermes` | 8192 | ~6 GB | Consumer GPUs (RTX 3050/4050/2060) |
| `llama-3.1-8b-instruct` | `meta-llama/Llama-3.1-8B-Instruct` | None (FP16) | `llama3_json` | 32768 | ~20 GB | Workstations / Cloud (RTX 3090, A10G) |
| `qwen2.5-7b-instruct` | `Qwen/Qwen2.5-7B-Instruct` | None (FP16) | `hermes` | 32768 | ~18 GB | Workstations (RTX 3090/4090, L4) |
| `qwen2.5-coder-32b-instruct-awq` | `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` | `awq` | `hermes` | 32768 | ~24 GB | High-End Workstations (RTX 4090, A100) |

---

## 8. Automated Test Suite & Reliability

The codebase includes an extensive automated test suite covering unit tests, API integration, streaming correctness, backpressure, and hardware validation:

```bash
# Run the full test suite
uv run pytest -q

# Run static type checking and linting
uv run mypy src
uv run ruff check .
```

| Component | Status | Description |
|---|---|---|
| **FastAPI Gateway Core & Routing** | Passed (140 tests) | OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints. |
| **Streaming Normalization Layer** | Passed | Strict OpenAI chunk compliance (metadata injection, finish reasons, guaranteed `[DONE]`). |
| **Client Disconnect Cancellation** | Passed | Detects client drop via `request.is_disconnected()`; commands vLLM to cancel generation. |
| **Concurrency Limiter (429 Backpressure)** | Passed | Rejects excess requests beyond `YLLM_MAX_CONCURRENT_REQUESTS` with HTTP 429 and `Retry-After`. |
| **Key Management (CLI & Hash Storage)** | Passed | Plaintext keys never stored; timing-safe SHA-256 HMAC lookup; CLI key management. |
| **Secret Redaction in Logs & Config** | Passed | Automatically masks Bearer tokens, API keys, and HF tokens from logs. |
| **Lifecycle State Machine** | Passed | 12 explicit states, atomic transitions, and bounded sliding-window restart supervisor. |
| **VRAM Memory Heuristic Estimator** | Passed | Computes advisory memory consumption ranges based on model architecture before launch. |

---

## 9. Troubleshooting

- **503 Service Unavailable (`{"status":"LOADING_MODEL","ready":false}`):**
  Server is downloading or loading weights into VRAM. Monitor logs with `docker logs -f yourllm`.
- **401 Unauthorized / Gated Repo Error:**
  Llama 3.1 requires license agreement. Set `HF_TOKEN=hf_...` in your `.env` file.
- **CUDA Out of Memory (OOM):**
  1. Reduce context length: `YLLM_MAX_MODEL_LEN=8192` (or `16384`).
  2. Lower utilization cap: `YLLM_GPU_MEMORY_UTILIZATION=0.85`.
  3. Switch to a smaller model (e.g., `qwen2.5-coder-1.5b-instruct`) or quantized profile (`awq`).
- **Coding Agent Times Out:**
  Ensure `YLLM_ENABLE_PREFIX_CACHING=true` is enabled to avoid recomputing large context prompts on multi-turn conversations.

---

## 10. Architecture Overview

```text
       Coding Agents (Cline / Roo Code / Continue)
                         |
                         | OpenAI-Compatible API (Bearer Auth)
                         v
         +-------------------------------+
         |       FastAPI Gateway         |
         |  - API Key Auth (HMAC Hash)   |
         |  - Concurrency Limiter (429)  |
         |  - Streaming Normalizer       |
         |  - Token & Latency Metrics    |
         +-------------------------------+
                         |
                         | Internal Localhost Loopback
                         v
         +-------------------------------+
         |     vLLM Subprocess Group     |
         |  - PagedAttention Engine      |
         |  - GPU KV Cache Management    |
         |  - Tool Call Parsers          |
         +-------------------------------+
```
