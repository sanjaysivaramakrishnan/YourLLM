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

## 4. Remote & Cloud Deployment (Accessing Compute from Your Laptop)

If your GPU runs on a remote cloud instance (RunPod, Lambda Labs, AWS, GCP) or on a friend's GPU machine / homelab PC, you do not need to install the engine on your personal laptop—you simply point your coding agent (Cline / Roo Code / Continue) to the remote server URL.

```mermaid
flowchart LR
    subgraph ClientLaptop ["Your Laptop (Client)"]
        Cline["Coding Agent (Cline / Roo Code)"]
    end

    subgraph Transit ["Network / Internet"]
        Direct["Option 1: Cloud Public IP / Proxy URL"]
        VPN["Option 2: Tailscale Mesh VPN"]
        Tunnel["Option 3: Ngrok / Cloudflare Tunnel"]
    end

    subgraph RemoteCompute ["GPU Host (Server / Friend's PC)"]
        Gateway["YourLLM Gateway (:8000)"]
        vLLM["vLLM Engine"]
        GPU["NVIDIA GPU"]
    end

    Cline --> Transit
    Transit --> Gateway --> vLLM --> GPU
```

### Scenario A: Cloud GPU Server (RunPod, Lambda Labs, AWS, GCP)
Cloud servers live directly on the internet with dedicated public IP addresses:

1. **Deploy the container on the cloud server:**
   ```bash
   docker run -d \
     --name yourllm \
     --gpus all \
     -p 8000:8000 \
     --shm-size=8g \
     -v yourllm-data:/data \
     --env-file .env \
     --restart unless-stopped \
     sanjaydocker832/yourllm:latest
   ```
2. **Retrieve the generated API key from server logs:**
   ```bash
   docker logs yourllm
   ```
3. **Configure Cline on your laptop:**
   - **API Provider:** `OpenAI Compatible`
   - **Base URL:** `http://<SERVER_PUBLIC_IP>:8000/v1`
   - **API Key:** `yllm_...`
   - **Model ID:** Match your configured `YLLM_SERVED_MODEL_NAME` (e.g. `llama-3.1-8b` or `qwen-coder-1.5b`)

> [!TIP]
> **RunPod One-Click Proxy:** When exposing HTTP port `8000` on RunPod, you get an automatic HTTPS proxy URL (`https://<pod-id>-8000.proxy.runpod.net/v1`). You can paste this URL directly into Cline from anywhere without configuring firewall rules or domain certificates!

---

### Scenario B: Connecting to Another Laptop / Homelab PC Across the Internet
Home laptops and homelabs sit behind residential Wi-Fi routers (NAT) and do not have public static IPs. Use one of these free tools to connect:

1. **Tailscale (Recommended — Private Encrypted Mesh):**
   - Install [Tailscale](https://tailscale.com) on both laptops (free).
   - Your GPU host gets a secure Tailscale IP (e.g. `100.85.24.12`).
   - In Cline on your laptop, configure Base URL: `http://100.85.24.12:8000/v1`.
   - Works automatically across homes, coffee shops, and mobile networks with zero router configuration.

2. **Ngrok / Cloudflare Tunnel (Instant Public HTTPS URL):**
   - Run a tunnel on the GPU host:
     ```powershell
     ngrok http 8000
     ```
   - Copy the generated HTTPS address (e.g., `https://a1b2.ngrok-free.app/v1`) into Cline on your laptop with your API key.

3. **Same Local Wi-Fi Network (LAN):**
   - On the GPU host, start Docker with `-p 0.0.0.0:8000:8000`.
   - Find the local IP on the GPU host via `ipconfig` (e.g. `192.168.1.45`).
   - In Cline on your laptop, configure Base URL: `http://192.168.1.45:8000/v1`.

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

## 7. Supported Models & Curated Profiles

### Running Any Open-Source Hugging Face Model
YourLLM can run **any open-source text or coding LLM on Hugging Face** supported by **vLLM v0.7.3**—you do not need a pre-written profile. Simply set `YLLM_MODEL=<org/model-name>` in `.env`.

**Supported model families include:**
- **Qwen 2.5 & Qwen 2.5 Coder** (1.5B, 3B, 7B, 14B, 32B, 72B)
- **Meta Llama 3 / 3.1 / 3.2 / 3.3** *(requires `HF_TOKEN` for gated Meta repositories)*
- **DeepSeek** (DeepSeek-R1-Distill, DeepSeek-Coder-V2)
- **Mistral / Mixtral / Codestral**, **Phi-3 / Phi-4**, **Gemma 2**, **IBM Granite**, **StarCoder2**
- **Quantization formats:** FP16/BF16, **AWQ**, **GPTQ**, and **FP8**

**Example — Running an Unprofiled Quantized Model on a 6 GB Laptop (`Qwen/Qwen2.5-3B-Instruct-AWQ`):**
```ini
YLLM_MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ
YLLM_SERVED_MODEL_NAME=qwen2.5-3b-awq
YLLM_QUANTIZATION=awq
YLLM_TOOL_CALL_PARSER=hermes
YLLM_CHAT_TEMPLATE=/opt/yourllm/templates/tool_chat_template_hermes.jinja
YLLM_GPU_MEMORY_UTILIZATION=0.90
YLLM_MAX_MODEL_LEN=16384
```

**Compatibility notes:**
- **Weight formats:** Uses Hugging Face **Safetensors** (FP16/BF16/AWQ/GPTQ/FP8)—not `.gguf` files (which are for llama.cpp/Ollama).
- **Tool calling:** Supported parsers for `YLLM_TOOL_CALL_PARSER`: `hermes`, `llama3_json`, `mistral`, `granite`, `pythonic`, `openai`.
- **Text-only:** Enforces text chat completions and tool calling (multimodal image/audio payloads are rejected).
- **Engine baseline:** Pinned to **vLLM v0.7.3** for CUDA 12.1 compatibility across wide NVIDIA driver ranges.

### Curated Model Profiles
For zero-config setup (`YLLM_PROFILE=auto`), YourLLM includes versioned profiles pairing popular coding models with their verified tool-call parsers and bundled chat templates in `src/yourllm/models/profiles/`:

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
