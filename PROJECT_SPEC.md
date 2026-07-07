# Anvil — Forge Anything.

**A transparent, modular, terminal-first local AI runtime.**

Anvil is a single-binary, zero-dependency local LLM tool that links `llama.cpp` directly at the native level. It is designed to be the fastest, most transparent, and most jank-free local AI tool in existence. It is the definitive Ollama alternative for developers who refuse slow defaults, opaque protocols, or resource waste顶hogs.

---

## 1. Core Philosophy

- **Janklessness above all.** A dropped TUI framerate, a 3-second startup delay, a JSON roundtrip per token — all of it is unacceptable. Every millisecond must be justified.
- **Transparent by default.** No hidden blob storage, no opaque protocols. Models are plain GGUF files. Configs are plain JSON. Every default is documented and overridable.
- **Everything is opt-in.** The user chooses what they want. Nothing is forced. Not the TUI, not the API server, not even TurboQuant. We suggest the best defaults, but the user decides.
- **Maximum speed, zero bloat.** Inference runs in-process via native C FFI. No subprocess spawn overhead. No HTTP serialization per token.
- **Terminal-first, human-friendly.** Drop a beginner into a TUI chat with one command. Give a pro total control via flags, config files, and a raw OpenAI-compatible API.
- **One binary.** The user downloads `anvil`. It works. No extra files, no runtime downloads for the core.

---

## 2. The Problem with the Status Quo

### Ollama
- Opaque blob storage, custom non-standard API, always-on daemon, slow defaults, closed telemetry, unoptimized inference.

### llama.cpp
- Brilliant but a nightmare to configure. The user must manually set quants, context size, GPU layers, backend, and flags. Beginners give up.

### Anvil bridges both worlds.
| Status Quo Problem | Anvil's Fix |
|---|---|
| Opaque storage | Models are plain GGUF files in `~/.anvil/models/` |
| Custom API | Drop-in OpenAI-compatible API (but in-process, not a subprocess) |
| Slow/unoptimized defaults | Hardware probe + auto-optimization on first run |
| Always-on daemon | Zero idle overhead. The binary runs when you tell it to. |
| Hard to configure | `anvil run llama3` — it just works at max speed |
| Resource hog | TurboQuant + speculative enabled by default where applicable |

---

## 3. Architecture

Anvil is a **single Rust binary** that compiles the `llama.cpp` C++ backend into itself as a static library. Inference is a native function call, not a subprocess.

### 3.1 The Core Engine (In-Process)

The `llama.cpp` codebase (`anvil-llama-turbo` fork) is compiled as a static Rust dependency at build time. The resulting `anvil` binary contains:

- The full `llama.cpp` inference engine (CPU, Metal, CUDA, Vulkan backends)
- TurboQuant KV cache compression
- MTP (Multi-Token Prediction) speculative decoding for Gemma 4
- NextN speculative decoding for Qwen 3.x
- All model architectures supported by upstream

**How:** Rust FFI via `include/llama.h` (pure C API). Zero serialization overhead.

### 3.2 The Orchestrator (Rust)

Pure Rust code that manages:
- **Model lifecycle:** Download, cache, load, unload
- **Inference loop:** Feed prompts, stream tokens, manage KV cache
- **Hardware probing:** Auto-detect CPU/GPU/RAM, select optimal settings
- **Configuration:** Plain JSON, fully editable
- **Optional components:** Only loaded if the user enables them

### 3.3 Optional Components (All Opt-In)

| Component | Trigger | What it does |
|---|---|---|
| **TUI** | `anvil run <model>` (default) or `anvil --tui` | ratatui-based chat, real-time stats, memory visualization |
| **API Server** | `anvil serve` | axum-based OpenAI-compatible HTTP server, in-process |
| **Cloud Proxy** | `cloud-proxy` feature enabled | Routes requests to OpenAI, Anthropic, Gemini |
| **Finetuning** | `anvil finetune` | Spawns self-contained training engine, streams progress, outputs GGUF |

### 3.4 Build-Time Modularity (Features)

Anvil uses Rust `features` for compile-time modularity:

```bash
cargo build --release # Full build with all features
cargo build --release --no-default-features --features "turbo,mcp" # Smaller binary
```

| Feature | Default | Description |
|---|---|---|
| `turbo` | yes | TurboQuant KV cache compression |
| `mtp` | yes | Multi-Token Prediction (Gemma 4) |
| `nextn` | yes | NextN speculative decoding (Qwen 3.x) |
| `tui` | yes | Terminal User Interface (ratatui) |
| `api` | yes | OpenAI-compatible HTTP server (axum) |
| `cloud` | no | Cloud provider proxy (OpenAI, Anthropic, etc.) |
| `finetune` | no | Training pipeline support |

Users who want a minimal binary:
```bash
cargo build --release --no-default-features --features "turbo"
```

---

## 4. The User Experience

### 4.1 Installation

```bash
curl -sSL https://raw.githubusercontent.com/gondaliyashreyan1/Anvil/main/install.sh | sh
```

The installer is a single POSIX-compliant shell script that downloads and installs Anvil with zero dependencies.

#### Quick Install (Default)
```bash
curl -sSL https://raw.githubusercontent.com/gondaliyashreyan1/Anvil/main/install.sh | sh
# → Detects OS/arch
# → Downloads latest release binary
# → Installs to ~/.anvil/bin/anvil
# → Optionally adds to PATH
# → Runs hardware probe
# → Done in < 10 seconds
```

#### Interactive TUI Install
```bash
curl -sSL https://raw.githubusercontent.com/gondaliyashreyan1/Anvil/main/install.sh | sh -s -- --interactive
```

Launches a beautiful terminal installer with:
- **OS/Arch Detection:** Shows detected platform with a checkmark.
- **Install Location:** Defaults to `~/.anvil/bin/`, user can change.
- **Feature Selection:** Checkboxes for optional components:
  - [x] Core Engine (required)
  - [x] TUI Chat Interface
  - [x] API Server
  - [x] TurboQuant Optimization
  - [ ] Cloud Proxy
  - [ ] Finetuning Tools
- **Theme Selector:** `forge` (default), `midnight`, `stealth`.
- **PATH Setup:** Add to `.zshrc`/`.bashrc` or skip.
- **Progress Bar:** Real-time download progress with speed display.

#### Silent/Non-Interactive Install
```bash
curl -sSL https://raw.githubusercontent.com/gondaliyashreyan1/Anvil/main/install.sh | sh -s -- --minimal       # Core only
curl -sSL https://raw.githubusercontent.com/gondaliyashreyan1/Anvil/main/install.sh | sh -s -- --full         # Everything
curl -sSL https://raw.githubusercontent.com/gondaliyashreyan1/Anvil/main/install.sh | sh -s -- --help        # See all flags
```

#### Installer Features
- **No sudo required.** Installs to user directory by default.
- **Atomic install.** Downloads to temp, verifies checksum, then swaps.
- **Rollback.** If install fails, nothing is left behind.
- **Self-contained.** The script itself is the only file downloaded. No curl dependencies beyond what's on every Unix system.
- **Windows support.** Powershell one-liner: `irm https://raw.githubusercontent.com/gondaliyashreyan1/Anvil/main/install.ps1 | iex`

#### What Happens After Install
1. Binary placed at `~/.anvil/bin/anvil`
2. `~/.anvil/config.json` created with defaults
3. Hardware probe runs once, writes profile to config
4. `anvil --version` works immediately
5. `anvil self-update` available for future updates

**Build from source (for tinkerers):**
```bash
git clone https://github.com/gondaliyashreyan1/Anvil
cd Anvil
cargo build --release
```

### 4.2 First Run: `anvil run <model>`

```bash
anvil run llama3.1
```

1. **Hardware Probe (runs once):** Detects CPU features (AVX2/AVX-512), GPU (VRAM, type), RAM.
2. **Auto-Optimization:** Selects the optimal profile:
   - **M4 MacBook** → Metal backend, `turbo3` KV, `Q4_K_M` quants, MTP enabled.
   - **RTX 4090** → CUDA backend, `turbo3` KV, `TQ4_1S` quants, NextN enabled.
   - **No GPU** → CPU backend, standard quants, no speculative.
3. **Download:** Fetches the best pre-quantized GGUF from HuggingFace if not cached.
4. **Launch:** TUI chat opens instantly.

To skip the TUI and just get a REPL:
```bash
anvil run llama3.1 --no-tui
```

### 4.3 Pro Mode: Full Control

```bash
anvil run mymodel.gguf \
  --backend cuda \
  --quant TQ4_1S \
  --ctx 131072 \
  --no-turboquant \
  --spec-type mtp \
  --ngl 99
```

### 4.4 The API Server (Opt-In)
```bash
anvil serve
# Opens OpenAI-compatible API on localhost:11434
# Calls llama_decode() directly. No subprocess. No serialization.
```

### 4.5 The Dashboard (Opt-In)
```bash
anvil dashboard
# Full-screen TUI for monitoring all running models, memory, tokens/sec
```

### 4.6 Finetuning (Opt-In)
```bash
anvil finetune --base models/llama3.gguf --data my_data.jsonl --epochs 3
```

---

## 5. Janklessness Manifesto

A feature is not complete until it meets these criteria:

| Criterion | Target |
|---|---|
| **Startup time** (from `anvil run` to first token) | `< 2 seconds` (after model downloaded) |
| **TUI framerate** | `60 fps` minimum |
| **Inference latency** | Zero serialization overhead (native call) |
| **Idle overhead** | Zero. Background threads sleep. |
| **Failure mode** | Single binary. Kill it, everything dies. |
| **Config time** | Zero. Auto-probed on first run. |
| **Update time** | `anvil self-update` — delta, <5 seconds. |

---

## 6. Directory Structure

```
~/.anvil/
  config.json              # Global user defaults (readable, editable)
  models/                  # Plain GGUF files
  cache/                   # HuggingFace download cache
  logs/                    # Plain text logs
```

The Anvil source tree:
```
Anvil/
├── backends/
│   └── llama-turbo/      # Submodule: anvil-llama-turbo (C++ backend)
├── src/
│   ├── main.rs           # Entrypoint
│   ├── lib.rs            # Public API
│   ├── tui/              # Terminal UI (ratatui, opt-in via 'tui' feature)
│   ├── api/              # OpenAI-compatible HTTP server (axum, opt-in via 'api' feature)
│   ├── engine/           # In-process llama.cpp wrapper
│   │   ├── mod.rs
│   │   ├── context.rs    # llama_context management
│   │   ├── model.rs      # llama_model management
│   │   ├── sampling.rs   # Token sampling
│   │   └── kv_cache.rs   # KV cache configuration
│   ├── hardware/         # Hardware probing + auto-optimization
│   │   ├── mod.rs
│   │   └── probe.rs
│   ├── model_manager/    # Download, cache, GGUF validation
│   │   ├── mod.rs
│   │   └── huggingface.rs
│   ├── cloud/            # Cloud proxy (opt-in via 'cloud' feature)
│   │   ├── mod.rs
│   │   ├── openai.rs
│   │   └── anthropic.rs
│   └── finetune/         # Training pipeline (opt-in via 'finetune' feature)
│       └── mod.rs
├── Cargo.toml            # Rust project + features definition
├── CMakeLists.txt        # Builds llama.cpp as static lib
└── build.rs              # Custom build script: links libllama.a
```

---

## 7. Key Features & Implementation Details

### 7.1 TurboQuant Integration
- **What:** WHT-rotated low-bit quantization for KV cache and weights.
- **How:** Native in `llama.cpp` core (our fork). Enabled at runtime via `llama_context` params.
- **Default:** `turbo3` for KV cache. Overridable via `--cache-type-k` / `--cache-type-v`.
- **Disable:** `--no-turboquant` or compile without the `turbo` feature.

### 7.2 MTP & NextN Speculative Decoding
- **MTP (Gemma 4):** Separate assistant head loaded alongside target model. Enabled for supported models via `LLAMA_SPECULATIVE` param.
- **NextN (Qwen 3.x):** Shared-model draft context. Enabled for `_MTP.gguf` models.
- **Both:** Configurable at runtime. No model changes needed.

### 7.3 Hardware-Adaptive Optimization
- **Probe:** `hwloc` + `cpuid` + OS APIs. Runs once, stores profile in `config.json`.
- **Profile:** Maps hardware to best backend (Metal/CUDA/Vulkan/CPU), quant level, context size, and speculative mode.
- **Override:** User can pin profiles or create custom ones.

### 7.4 The API Server (In-Process)
- **Framework:** `axum` (tokio-based).
- **Endpoints:** Full OpenAI compatibility:
  - `GET /v1/models`
  - `POST /v1/chat/completions` (streaming + non-streaming)
  - `POST /v1/completions`
  - `POST /v1/embeddings`
  - `GET /health`
- **Latency:** Calls `llama_decode()` directly. No HTTP overhead for inference.

### 7.5 The TUI (ratatui)
- **Layout:** Multi-pane. Chat, memory bars, performance metrics, logs.
- **Modes:**
  - **Chat:** Inline chat with markdown rendering.
  - **Monitor:** k9s-style model list. j/k to navigate, d to unload.
  - **Bench:** Live benchmark graphs.
- **Keymaps:** Vim-style (`hjkl`, `?` for help, `:` for command palette).

---

## 8. Design Principles

1. **Janklessness is non-negotiable.** If it's slow, it doesn't ship.
2. **Opt-in, not opt-out.** Every feature beyond the core engine is optional at compile time or runtime.
3. **No hidden magic.** Auto-selected settings are visible and overridable.
4. **Files are plain.** Models are files. Configs are JSON. Logs are text.
5. **One binary.** The user downloads `anvil`. The end.

---

## 9. MVP Roadmap

### Phase 1: The Core Runner
- [ ] Rust project scaffold with `llama.cpp` static dependency
- [ ] FFI bindings for `llama.h` (model load, decode, sampling)
- [ ] `anvil run <model>` with in-process inference and simple REPL
- [ ] Hardware prober (OS, CPU, GPU, RAM)

### Phase 2: The TUI
- [ ] ratatui-based chat interface
- [ ] Real-time memory/performance visualization
- [ ] Model list / monitor mode

### Phase 3: The API Server
- [ ] axum OpenAI-compatible server
- [ ] Streaming and non-streaming completions
- [ ] In-process token generation (no serialization)

### Phase 4: Smarts
- [ ] Auto-optimizer based on hardware probe
- [ ] Feature-gated compilation (`--no-default-features`)
- [ ] Self-updater (`anvil self-update`)

### Phase 5: Power Tools
- [ ] Cloud proxy (OpenAI, Anthropic)
- [ ] Finetuning pipeline
- [ ] Cross-compilation CI (macOS Intel/ARM, Linux x64, Windows x64)

---

## 10. Brand

- **Name:** Anvil
- **CLI:** `anvil`
- **Tagline:** Forge anything.
- **Color Identity:** Deep charcoal, steel grey, forge-orange. Industrial, solid, no-bullshit.

---

*Anvil: the anvil doesn't ask questions. It just works.*
