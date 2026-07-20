#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <functional>
#include <cstdint>
#include <optional>

namespace anvil {

// ─── Configuration structures ───────────────────────────────────────────────

struct EngineConfig {
    std::string model_path;
    std::string mtp_head_path;       // Gemma 4 assistant GGUF
    std::string draft_model_path;    // NextN combined GGUF (can be same as model_path)

    int32_t n_gpu_layers     = -1;   // -1 = auto-detect
    int32_t n_gpu_layers_draft = -1;
    int32_t n_ctx            = 8192;
    int32_t n_batch          = 512;
    int32_t n_threads        = 0;    // 0 = auto
    int32_t n_threads_batch  = 0;

    float   temperature      = 0.8f;
    int32_t top_k            = 40;
    float   top_p            = 0.95f;
    float   min_p            = 0.05f;
    int32_t max_tokens       = -1;   // -1 = unlimited

    std::string cache_type_k = "turbo3";
    std::string cache_type_v = "turbo3";

    bool    flash_attn       = false;
    bool    embeddings       = false;
    bool    use_mmap         = true;

    // Speculative decoding
    std::string spec_type    = "none"; // "none", "mtp", "nextn", "draft-simple"
    int32_t draft_block_size = 4;
    int32_t draft_max        = 16;
    int32_t draft_min        = 0;

    // System prompt
    std::string system_prompt;
};

// ─── Inference result ───────────────────────────────────────────────────────

struct GenerateResult {
    std::string text;
    int32_t tokens_generated = 0;
    float   prompt_eval_ms  = 0.0f;
    float   eval_ms         = 0.0f;
    float   tokens_per_sec  = 0.0f;
};

// ─── Progress callback ──────────────────────────────────────────────────────

using ProgressCallback = std::function<bool(float progress)>;

// ─── Engine ─────────────────────────────────────────────────────────────────

class Engine {
public:
    Engine() = default;
    ~Engine();

    // Non-copyable, movable
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;

    // Initialize the llama backend (call once per process)
    static void backend_init();
    static void backend_free();

    // Explicit cleanup (safe to call multiple times, idempotent)
    // Must be called before backend_free() to avoid Metal cleanup crash
    void cleanup();

    // Load model from file
    bool load_model(const EngineConfig& config, ProgressCallback progress = nullptr);

    // Load MTP assistant head (Gemma 4)
    bool load_mtp_assistant(const std::string& path);

    // Create inference context
    bool create_context(const EngineConfig& config);

    // Tokenize text
    std::vector<llama_token> tokenize(const std::string& text, bool add_special = false);

    // Detokenize tokens to text
    std::string detokenize(const llama_token* tokens, int32_t n_tokens);

    // Generate text from prompt
    GenerateResult generate(
        const std::string& prompt,
        const EngineConfig& config,
        std::function<void(const std::string& token)> on_token = nullptr
    );

    // Single decode step
    int32_t decode(const llama_batch& batch);

    // Sample next token
    llama_token sample();

    // Get logits for last token
    float* get_logits();

    // Get embeddings for last token
    float* get_embeddings();

    // Clear KV cache
    void kv_clear();

    // Get model info
    uint64_t model_size() const;
    uint64_t model_params() const;
    int32_t n_ctx() const;
    int32_t n_vocab() const;

    // Check if model is loaded
    bool is_loaded() const { return model_ != nullptr; }
    bool has_context() const { return ctx_ != nullptr; }
    bool has_mtp() const { return mtp_assistant_ != nullptr; }

private:
    UniqueModel model_;
    UniqueModel mtp_assistant_;
    UniqueContext ctx_;
    UniqueSampler sampler_;

    const llama_vocab* vocab_ = nullptr;

    // Pre-allocated buffers to avoid per-token allocations
    TokenBuffer token_buf_;
    std::string detok_buf_;

    // Progress callback storage (prevents dangling pointer from stack)
    ProgressCallback progress_cb_;

    bool embeddings_enabled_ = false;
};

} // namespace anvil
