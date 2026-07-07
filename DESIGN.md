# Anvil v0.1 - Full-Scale How-It-Works Report

## Overview

Anvil v0.1 is a **single binary** that embeds the entire llama.cpp inference engine via in-process C FFI. The user sees a friendly CLI (`anvil run model.gguf --temp 0.7`), but under the hood we call llama.cpp's C API directly via `llama.h`. Zero subprocess. Zero serialization. Maximum speed.

---

## 1. Build Pipeline: How the Single Binary is Produced

```
Developer runs: cargo build --release

1. build.rs (Rust build script)
   |
   |-- Calls cmake on backends/llama-turbo/
   |   |
   |   +-- Produces libllama.a (static library)
   |
   +-- Links libllama.a + ggml libs into the Rust binary
   +-- src/engine/raw.rs (raw FFI declarations)

2. cargo compiles Rust sources
   |
   +-- src/main.rs (CLI parsing, dispatch)
   +-- src/engine/mod.rs (safe Rust wrappers, inference loop)
   +-- src/engine/raw.rs (extern "C" blocks for llama.h)
   +-- src/hardware/probe.rs (hardware detection)
   +-- src/config.rs (JSON config read/write)

3. Linker produces final anvil binary
   |
   +-- Contains libllama.a (all backends: Metal, CUDA, Vulkan, CPU)
   +-- Contains Rust orchestrator with our own inference loop
```

### 1.1 Key Build Details

build.rs responsibilities:
- Detect host triple (x86_64-apple-darwin, aarch64-apple-darwin, etc.)
- Set CMAKE_OSX_DEPLOYMENT_TARGET to auto-detected macOS version
- Enable backend-specific flags for cmake:
  - macOS: -DLLAMA_METAL=ON
  - Linux with NVIDIA: -DLLAMA_CUDA=ON
  - Linux generic: -DLLAMA_VULKAN=ON or CPU only
- Build ONLY libllama.a (no main.cpp, no examples)
- Emit cargo:rustc-link-search and cargo:rustc-link-lib for all static libs
- Link required frameworks: Metal, Foundation, Accelerate, QuartzCore, CoreFoundation
- Link C++ standard library: -lc++
- Link Objective-C runtime: -lobjc

### 1.2 Why NOT main.cpp Passthrough

Early design considered patching main.cpp to export anvil_llama_main(). This was rejected because:
- It creates an dependency on llama.cpp's CLI argument parsing
- It couples us to their main loop timing and output handling
- It makes it impossible to add our own TUI, streaming, or API layers later
- It's more complex, not less

Instead, we call llama.h C API directly and write our own inference loop in Rust.

---

## 2. Runtime Data Flow: CLI to Tokens

```bash
$ anvil run llama3.1.gguf --temp 0.7 --ctx 4096
```

```
Step 1: Rust CLI Parsing (src/main.rs)
 + clap parses friendly flags
 + Friendly flag --ctx maps to --ctx-size
 + --temp maps to --temperature
 + --ngl absent? query hardware prober
 + Resolve final flags

Step 2: Hardware Prober (if --ngl not specified)
 + Detect macOS + Apple Silicon -> --ngl 99, --backend metal
 + Detect NVIDIA GPU -> --ngl 99, --backend cuda
 + Detect AMD GPU -> --ngl 99, --backend hip
 + No GPU detected -> --ngl 0

Step 3: Config Overlay
 + Read ~/.anvil/config.json
 + Merge CLI flags (CLI overrides config)
 + Write missing fields back to config

Step 4: Model Load (Rust -> C FFI)
 + llama_backend_init()
 + llama_model_load_from_file(path, params)
 + llama_init_from_model(model, ctx_params)

Step 5: Inference Loop (Rust -> C FFI)
 + llama_tokenize() to encode prompt
 + llama_decode() to process prompt
 + While not done:
   + llama_get_logits_ith() for last token
   + llama_sampler_sample() to pick next token
   + llama_token_to_piece() to convert to string
   + Print token to stdout (streaming)
   + llama_decode() the new token

Step 6: Cleanup
 + llama_free(ctx)
 + llama_model_free(model)
 + llama_backend_free()

Step 7: Rust Exit
 + Return llama.cpp's exit code
```

---

## 3. Flag Parsing and Routing

### 3.1 Friendly Flags (Our Aliases)

These are NOT llama.cpp flags. We map them before calling the C API.

| Friendly Flag | Maps To | Default |
|---|---|---|
| --ctx <n> | ctx_size in llama_context_params | auto-detect |
| --temp <f> | temperature in sampler chain | 0.8 |
| --max-tokens <n> | max generation tokens | -1 (unlimited) |
| --system "..." | prepended to prompt with chat template | (none) |
| --prompt "..." | raw prompt text | (required if not interactive) |
| --interactive | enable REPL mode | false |
| --ngl <n> | n_gpu_layers in llama_model_params | auto-detect |
| --flash-attn | flash_attn_type in context params | false (auto for Metal) |
| --backend <name> | backend hint (for prober) | auto-detect |

### 3.2 Raw llama.cpp Flags (Passthrough)

Any flag that doesn't match a friendly alias is passed through verbatim to llama.cpp's parse_args(). Examples:

```bash
anvil run model.gguf --rope-scaling yarn --yarn-ext-factor 1.0
# passes --rope-scaling and --yarn-ext-factor directly to main.cpp

anvil run model.gguf --no-context-shift
# passes --no-context-shift directly to main.cpp
```

### 3.3 Flag Resolution Rules

1. Friendly aliases are resolved first. They become their llama.cpp equivalents.
2. Unknown flags are passed through as-is. No validation from our side.
3. llama.cpp validates all flags. If there is a conflict (e.g., --ctx 4096 and --ctx-size 8192), llama.cpp's parse_args() handles it. Typically: last one wins, or error if the value is truly invalid.
4. If llama.cpp exits with error, Anvil propagates it. Zero magic, zero swallowing of errors.

### 3.4 Auto-Detection Rules

When --ngl is NOT specified by the user:

| Detected Hardware | --ngl | --backend | Notes |
|---|---|---|---|
| Apple Silicon (M1-M4) | 99 | metal | Full offload |
| Apple Intel (iGPU) | 0 | metal | Metal but no GPU memory |
| NVIDIA GPU (any) | 99 | cuda | Full offload |
| AMD GPU (ROCm/HIP) | 99 | hip | Full offload |
| Intel Arc (SYCL) | 99 | sycl | Full offload |
| No GPU | 0 | cpu | CPU backend |

When --ngl IS specified, we use the user's value and do not override.

### 3.5 TurboQuant KV Cache Defaults

Anvil defaults to TurboQuant3 for both K and V caches when the user does not specify --cache-type-k or --cache-type-v.

| Cache | Default | Rationale |
|---|---|---|
| K (key)   | turbo3 | ~4.3x compression over f16, near-lossless quality |
| V (value) | turbo3 | ~4.3x compression over f16, near-lossless quality |

Turbo3 compresses the KV cache to ~3 bits per value using a WHT-rotated, two-stage codebook process. Perplexity stays almost identical to full f16, making it effectively near-lossless for practical workloads. At 2-bit (turbo2), quality dips become measurable; at 3-bit (turbo3), the tradeoff is negligible.

Override at your own risk:
```bash
anvil run model.gguf --cache-type-k f16 --cache-type-v f16
```

---

## 4. Hardware Prober

### 4.1 What We Detect

```rust
struct HardwareProfile {
    os: String,                    // macOS, Linux, Windows
    arch: String,                  // x86_64, aarch64
    cpu: String,
    cpu_features: Vec<String>,     // "avx2", "avx512", "neon"
    ram_gb: u64,                   // Total system RAM
    gpus: Vec<GPUInfo>,             // All detected GPUs
    best_backend: String,           // metal, cuda, hip, sycl, vulkan, cpu
    recommended_ngl: i32,          // Layers to offload
    recommended_ctx: u32,          // Default context size
}

struct GPUInfo {
    name: String,
    vendor: String,                // Apple, NVIDIA, AMD, Intel
    vram_mb: u64,                  // VRAM in MB
    is_discrete: bool,             // true for dGPU, false for iGPU
}
```

### 4.2 How Detection Works (Platform-specific)

macOS:
- sw_vers -productVersion -> OS version
- sysctl -n machdep.cpu.brand_string -> CPU model
- sysctl -n hw.memsize -> RAM
- sysctl -n hw.optional.arm64 -> ARM64 check
- sysctl -n hw.ncpu -> Core count
- Metal: always available on macOS
- GPU name: IOServiceMatching("IOGPU") or system_profiler SPDisplaysDataType

Linux:
- /proc/cpuinfo -> CPU features (AVX2, AVX-512, etc.)
- /proc/meminfo -> RAM
- nvidia-smi -> NVIDIA GPU info
- rocminfo / clinfo -> AMD/Intel GPU info
- vulkaninfo -> Vulkan-compatible GPUs
- lspci -> fallback GPU detection

Windows:
- GetSystemInfo() -> CPU/arch
- GlobalMemoryStatusEx() -> RAM
- EnumDisplayDevices() + nvml -> GPU info

### 4.3 Prober Output (First Run)

```
$ anvil run llama3.1.gguf
[Anvil] Detected hardware:
  OS:        macOS 15.1 (Apple Silicon)
  CPU:       Apple M4 Max (14 cores)
  RAM:       36 GB
  GPU:       Apple M4 Max 32-core GPU (24 GB unified)
  Backend:   Metal (auto-selected)
  Offload:   99 layers (full)
  Context:   8192

[llama.cpp] prompt: ...
Hello! How can I help you today?
```

Probe result is cached in ~/.anvil/config.json.

---

## 5. FFI Layer: The Exact Functions

### 5.1 Raw FFI (Rust extern "C" blocks)

```rust
// src/engine/raw.rs
use std::os::raw::{c_char, c_float, c_void};

#[repr(C)]
pub struct llama_model { _private: [u8; 0] }

#[repr(C)]
pub struct llama_context { _private: [u8; 0] }

#[repr(C)]
pub struct llama_vocab { _private: [u8; 0] }

#[repr(C)]
pub struct llama_sampler { _private: [u8; 0] }

pub type llama_token = i32;
pub type llama_pos = i32;

// ... structs for llama_model_params, llama_context_params, llama_batch ...

extern "C" {
    pub fn llama_backend_init();
    pub fn llama_backend_free();
    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_context_default_params() -> llama_context_params;
    pub fn llama_model_load_from_file(path: *const c_char, ...) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);
    pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
    pub fn llama_init_from_model(model: *mut llama_model, ...) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);
    pub fn llama_tokenize(vocab: *const llama_vocab, text: *const c_char, ...) -> i32;
    pub fn llama_token_to_piece(vocab: *const llama_vocab, token: llama_token, ...) -> i32;
    pub fn llama_n_vocab(vocab: *const llama_vocab) -> i32;
    pub fn llama_vocab_eos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch;
    pub fn llama_batch_free(batch: llama_batch);
    pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> i32;
    pub fn llama_get_logits_ith(ctx: *mut llama_context, i: i32) -> *mut c_float;
    pub fn llama_sampler_init_greedy() -> *mut llama_sampler;
    pub fn llama_sampler_init_dist(seed: u32) -> *mut llama_sampler;
    pub fn llama_sampler_accept(smpl: *mut llama_sampler, token: llama_token);
    pub fn llama_sampler_sample(smpl: *mut llama_sampler, ctx: *mut llama_context, idx: i32) -> llama_token;
    pub fn llama_sampler_free(smpl: *mut llama_sampler);
    pub fn llama_set_n_threads(ctx: *mut llama_context, n_threads: i32, n_threads_batch: i32);
    pub fn llama_print_system_info() -> *const c_char;
}
```

### 5.2 The Inference Loop

Our Rust code directly calls the C API:

```rust
// Pseudocode for src/engine/mod.rs

pub fn run_inference(model_path: &str, prompt: &str, max_tokens: i32) {
    unsafe { llama_backend_init(); }

    let model = load_model(model_path);
    let ctx = create_context(&model);

    let tokens = tokenize(ctx.vocab, prompt);
    decode_tokens(&ctx, &tokens);

    for _ in 0..max_tokens {
        let next_token = sample_next(ctx);
        if is_eos(next_token) { break; }
        let piece = token_to_string(ctx.vocab, next_token);
        print!("{}", piece); // streaming output
    }

    free_context(ctx);
    free_model(model);
    unsafe { llama_backend_free(); }
}
```

---

## 6. Config System

### 6.1 Config File: ~/.anvil/config.json

```json
{
  "version": "0.1.0",
  "hardware": {
    "probed_at": "2025-07-07T10:33:00Z",
    "os": "macos",
    "arch": "aarch64",
    "cpu": "Apple M4 Max",
    "ram_gb": 36,
    "gpus": [
      {
        "name": "Apple M4 Max 32-core GPU",
        "vendor": "Apple",
        "vram_mb": 24576,
        "is_discrete": false
      }
    ],
    "best_backend": "metal"
  },
  "defaults": {
    "ngl": 99,
    "ctx_size": 8192,
    "temperature": 0.8,
    "flash_attn": true
  },
  "paths": {
    "models_dir": "~/.anvil/models",
    "cache_dir": "~/.anvil/cache"
  }
}
```

### 6.2 Config Resolution Order (Highest to Lowest Priority)

1. CLI flags (anvil run --temp 0.3)
2. Config JSON (~/.anvil/config.json)
3. Hardware defaults (prober result)
4. Hardcoded sensible defaults (in src/config.rs)

---

## 7. Error Handling Strategy

### 7.1 Our Errors (Rust side, before calling C API)

| Error | Message | Action |
|---|---|---|
| Model file not found | "Model not found: ~/.anvil/models/llama3.1.gguf" | Exit code 1 |
| Hardware probe failed | "Could not detect GPU. Fallback to CPU." | Warn, continue --ngl 0 |
| Invalid model format | "File is not a valid GGUF" | Exit code 2 |
| Config JSON corrupt | "Config corrupted. Delete ~/.anvil/config.json and retry." | Exit code 3 |

### 7.2 llama.cpp Errors (propagated from C API)

| Error | Message | Action |
|---|---|---|
| Invalid flag value | (from llama.cpp stderr) | Print to stderr, exit with llama.cpp's return code |
| Out of memory | (from llama.cpp stderr) | Print to stderr, exit code 1 |
| Model load failure | (from llama.cpp stderr) | Print to stderr, exit code 1 |

### 7.3 Flag Conflict Handling

If a user does: anvil run model.gguf --ctx 4096 --ctx-size 8192

Both are passed to llama.cpp. parse_args() in main.cpp will see both. Typically: last one wins. If the flag is truly invalid, main.cpp will print an error and exit with non-zero code, which we propagate.

We do NOT pre-validate flags. We trust llama.cpp's parser.

---

## 8. File Structure (v0.1)

```
Anvil/
├── backends/
│   └── llama-turbo/                       # Submodule (our fork with TurboQuant)
│       ├── CMakeLists.txt
│       ├── include/
│       │   └── llama.h                    # C API we call via FFI
│       ├── src/                           # llama.cpp core
│       ├── common/                        # llama.cpp common lib
│       └── ggml/                          # ggml backend
│
├── src/
│   ├── main.rs                            # CLI entry, clap parsing, dispatch
│   ├── lib.rs                             # Public API re-exports
│   ├── cli.rs                             # Flag definitions, friendly to raw mapping
│   ├── config.rs                          # ~/.anvil/config.json read/write
│   ├── engine/
│   │   ├── mod.rs                         # Inference loop (tokenize, decode, sample)
│   │   └── raw.rs                         # Raw FFI declarations (extern "C" blocks)
│   └── hardware/
│       ├── mod.rs
│       └── probe.rs                       # Platform-specific hardware detection
│
├── Cargo.toml
├── build.rs                               # Compiles libllama.a via cmake, links everything
├── DESIGN.md                              # This file
├── PROJECT_SPEC.md                        # Full project spec
└── README.md
```

---

## 9. Build Dependencies

| Tool | Minimum Version | Purpose |
|---|---|---|
| Rust | 1.80+ | Anvil itself |
| CMake | 3.22+ | Build llama.cpp |
| C++ compiler | clang++ 14+ / g++ 12+ | Compile llama.cpp |
| Python | 3.10+ | llama.cpp build scripts |

Build commands:

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/gondaliyashreyan1/Anvil
cd Anvil

# Build everything (backend + Rust)
cargo build --release

# Result:
# target/release/anvil (single binary, fully self-contained)
```

---

## 10. What v0.1 Is NOT

| Feature | Status | Phase |
|---|---|---|
| TUI (ratatui) | Not in v0.1 | Phase 2 |
| API server (axum) | Not in v0.1 | Phase 3 |
| Cloud proxy | Not in v0.1 | Phase 5 |
| Finetuning | Not in v0.1 | Phase 5 |
| Model downloader | Not in v0.1 | v0.2 |
| Auto-optimizer (beyond --ngl) | Not in v0.1 | Phase 4 |
| Speculative decoding UI | Not in v0.1 | v0.2 |

v0.1 is the core engine. Everything else layers on top.

---

Anvil v0.1: one binary, zero overhead, all the power of llama.cpp, controlled by us.
