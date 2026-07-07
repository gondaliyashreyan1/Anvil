# Anvil v0.1 - Full-Scale How-It-Works Report

## Overview

Anvil v0.1 is a **single binary** that embeds the entire llama.cpp inference engine via in-process womb-to-tomb C FFI. The user sees a friendly CLI (`anvil run model.gguf --temp 0.7`), but under the hood it is a native function call into llama.cpp's `main.cpp` entrypoint. Zero subprocess. Zero serialization. Maximum speed.

---

## 1. Build Pipeline: How the Single Binary is Produced

```
Developer runs: cargo build --release

1. build.rs (Rust build script)
   |
   |-- Calls cmake on backends/llama-turbo/
   |   |
   |   +-- Produces libllama.a (static library)
   |   +-- Produces main.o (from our patched examples/main/main.cpp)
   |
   +-- Links libllama.a + main.o into the Rust binary
   +-- Generates src/engine/bindings.rs (raw FFI declarations)

2. cargo compiles Rust sources
   |
   +-- src/main.rs (CLI parsing, dispatch)
   +-- src/engine/mod.rs (safe Rust wrappers)
   +-- src/hardware/probe.rs (hardware detection)
   +-- src/config.rs (JSON config read/write)

3. Linker produces final anvil binary
   |
   +-- Contains libllama.a (all backends: Metal, CUDA, Vulkan, CPU)
   +-- Contains patched main.o (anvil_llama_main entrypoint)
   +-- Contains Rust orchestrator
```

### 1.1 Key Build Details

build.rs responsibilities:
- Detect host triple (x86_64-apple-darwin, aarch64-apple-darwin, etc.)
- Set CMAKE_OSX_ARCHITECTURES on macOS cross-compilation
- Enable backend-specific flags for cmake:
  - macOS: -DLLAMA_METAL=ON
  - Linux with NVIDIA: -DLLAMA_CUDA=ON
  - Linux generic: -DLLAMA_VULKAN=ON or CPU only
- Compile examples/main/main.cpp with anvil_llama_main() instead of main()
- Emit cargo:rustc-link-lib=static=llama and search paths

Why patch main.cpp?
Because main.cpp already implements EVERYTHING: argument parsing, tokenization, chat templates, the full decode loop, all samplers, all output formatting. By renaming main() to anvil_llama_main(), we get all of that for free. No reimplementation. No missing features.

---

## 2. Runtime Data Flow: CLI to Tokens

```bash
$ anvil run llama3.1.gguf --temp 0.7 --ctx 4096
```

```
Step 1: Rust CLI Parsing (src/main.rs)
 + clap derives parse args
 + Friendly flag --ctx maps to --ctx-size
 + --temp maps to --temperature
 + --ngl absent? query hardware prober
 + Build final argv: ["--model", "llama3.1.gguf",
                     "--temperature", "0.7",
                     "--ctx-size", "4096",
                     "--ngl", "33"]

Step 2: Hardware Prober (if --ngl not specified)
 + Detect macOS + Apple Silicon -> --ngl 99, --backend metal
 + Detect NVIDIA GPU -> --ngl 99, --backend cuda
 + Detect AMD GPU -> --ngl 99, --backend hip
 + No GPU detected -> --ngl 0

Step 3: Config Overlay
 + Read ~/.anvil/config.json
 + Merge CLI flags (CLI overrides config)
 + Write missing fields back to config

Step 4: FFI Invocation
 + Call anvil_llama_main(argc, argv) from Rust
   (defined in our patched main.cpp)

Step 5: llama.cpp Native Execution
 + parse_args() inside main.cpp handles all flags
 + llama_model_load_from_file(model, params)
 + llama_init_from_model(model, ctx_params)
 + Tokenize prompt, llama_decode() loop
 + Sample via llama_sampler_sample()
 + Stream tokens to stdout (raw, unbuffered)

Step 6: Cleanup
 + llama_free(ctx)
 + llama_model_free(model)
 + Return control to Rust

Step 7: Rust Exit
 + Print perf stats (optional)
 + Exit with llama.cpp's return code
```

---

## 3. Flag Parsing and Routing

### 3.1 Friendly Flags (Our Aliases)

These are NOT llama.cpp flags. We map them before passing the argv array.

| Friendly Flag | Maps To | Default |
|---|---|---|
| --ctx <n> | --ctx-size <n> | auto-detect |
| --temp <f> | --temperature <f> | 0.8 |
| --max-tokens <n> | --n-predict <n> | -1 (unlimited) |
| --system "..." | prepended to prompt with chat template | (none) |
| --prompt "..." | --prompt "..." | (required if not interactive) |
| --interactive | --interactive-first | false |
| --ngl <n> | --ngl <n> | auto-detect |
| --flash-attn | --flash-attn | false (auto for Metal) |
| --backend <name> | backend hint (for prober) | auto-detect |

### 3.2 Raw llama.cpp Flags (Passthrough)

Any flag that does not match a friendly alias is passed through verbatim to llama.cpp's parse_args(). Examples:

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

Anvil defaults to TurboQuant3 for both K and V caches when the user does not specify `--cache-type-k` or `--cache-type-v`.

| Cache | Default | Rationale |
|---|---|---|
| K (key)   | `turbo3` | ~4.3x compression over f16, near-lossless quality |
| V (value) | `turbo3` | ~4.3x compression over f16, near-lossless quality |

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
    os: OSType,                    // macOS, Linux, Windows
    arch: Architecture,            // x86_64, aarch64
    cpu_vendor: String,
    cpu_features: Vec<String>,     // "avx2", "avx512", "neon"
    ram_gb: u64,                   // Total system RAM
    gpu: Vec<GPUInfo>,             // All detected GPUs
    best_backend: Backend,         // metal, cuda, hip, sycl, vulkan, cpu
    recommended_ngl: i32,          // Layers to offload
    recommended_ctx: u32,          // Default context size
}

struct GPUInfo {
    name: String,
    vendor: GPUVendor,             // Apple, NVIDIA, AMD, Intel
    vram_mb: u64,                  // VRAM in MB
    is_discrete: bool,             // true for dGPU, false for iGPU
}
```

### 4.2 How Detection Works (Platform-specific)

macOS:
- sysctl -n hw.model -> CPU model
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

We declare only the C functions we need from llama.h:

```rust
// src/engine/raw.rs
use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
pub struct llama_model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_context {
    _private: [u8; 0],
}

extern "C" {
    pub fn llama_backend_init();
    pub fn llama_backend_free();
    pub fn llama_model_load_from_file(path: *const c_char, ...) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);
    pub fn llama_init_from_model(model: *mut llama_model, ...) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);
    pub fn llama_model_chat_template(model: *const llama_model, name: *const c_char) -> *const c_char;
    pub fn llama_chat_apply_template(tmpl: *const c_char, ...) -> i32;
    pub fn llama_print_system_info() -> *const c_char;
}
```

### 5.2 The anvil_llama_main Entrypoint

Our fork's examples/main/main.cpp gets this patch:

```cpp
// examples/main/main.cpp (PATCHED)
extern "C" int anvil_llama_main(int argc, char ** argv) {
    // ... original main.cpp body ...
    // Uses llama.cpp's parse_args(), the full decode loop, etc.
    // Just the function name changed from main to anvil_llama_main
    // and it is wrapped in extern "C"
}
```

Rust calls it:

```rust
// src/engine/entrypoint.rs
extern "C" {
    fn anvil_llama_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
}

pub fn run_llama_main(args: Vec<String>) -> i32 {
    // Convert to C strings, call anvil_llama_main
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

### 7.1 Our Errors (Rust side, before calling anvil_llama_main)

| Error | Message | Action |
|---|---|---|
| Model file not found | "Model not found: ~/.anvil/models/llama3.1.gguf" | Exit code 1 |
| Hardware probe failed | "Could not detect GPU. Fallback to CPU." | Warn, continue --ngl 0 |
| Invalid model format | "File is not a valid GGUF" | Exit code 2 |
| Config JSON corrupt | "Config corrupted. Delete ~/.anvil/config.json and retry." | Exit code 3 |

### 7.2 llama.cpp Errors (propagated from anvil_llama_main)

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
│       └── examples/
│           └── main/
│               └── main.cpp               # PATCHED: exports anvil_llama_main()
│
├── src/
│   ├── main.rs                            # CLI entry, clap parsing, dispatch
│   ├── lib.rs                             # Public API re-exports
│   ├── cli.rs                             # Flag definitions, friendly to raw mapping
│   ├── config.rs                          # ~/.anvil/config.json read/write
│   ├── engine/
│   │   ├── mod.rs                         # Safe wrappers (Model, Context, etc.)
│   │   ├── raw.rs                         # Raw FFI declarations
│   │   └── entrypoint.rs                  # Calls anvil_llama_main()
│   └── hardware/
│       ├── mod.rs
│       └── probe.rs                       # Platform-specific hardware detection
│
├── Cargo.toml
├── build.rs                               # Compiles llama.cpp + main.cpp, links everything
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

Anvil v0.1: one binary, zero overhead, all the power of llama.cpp.
