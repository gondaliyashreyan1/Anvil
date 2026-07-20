#include "anvil/engine.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <thread>
#include <atomic>

namespace anvil {

// ─── Backend lifecycle ──────────────────────────────────────────────────────

static bool g_backend_initialized = false;

void Engine::backend_init() {
    if (!g_backend_initialized) {
        llama_backend_init();
        g_backend_initialized = true;
    }
}

void Engine::backend_free() {
    if (g_backend_initialized) {
        llama_backend_free();
        g_backend_initialized = false;
    }
}

// ─── RAII cleanup ───────────────────────────────────────────────────────────

Engine::~Engine() {
    cleanup();
}

Engine::Engine(Engine&& o) noexcept
    : model_(std::move(o.model_))
    , mtp_assistant_(std::move(o.mtp_assistant_))
    , ctx_(std::move(o.ctx_))
    , sampler_(std::move(o.sampler_))
    , vocab_(o.vocab_)
    , token_buf_(std::move(o.token_buf_))
    , detok_buf_(std::move(o.detok_buf_))
    , embeddings_enabled_(o.embeddings_enabled_)
{
    o.vocab_ = nullptr;
    o.embeddings_enabled_ = false;
}

Engine& Engine::operator=(Engine&& o) noexcept {
    if (this != &o) {
        cleanup();
        model_ = std::move(o.model_);
        mtp_assistant_ = std::move(o.mtp_assistant_);
        ctx_ = std::move(o.ctx_);
        sampler_ = std::move(o.sampler_);
        vocab_ = o.vocab_;
        token_buf_ = std::move(o.token_buf_);
        detok_buf_ = std::move(o.detok_buf_);
        embeddings_enabled_ = o.embeddings_enabled_;
        o.vocab_ = nullptr;
        o.embeddings_enabled_ = false;
    }
    return *this;
}

void Engine::cleanup() {
    // Destroy in reverse order of creation
    sampler_.reset();
    ctx_.reset();
    mtp_assistant_.reset();
    model_.reset();
    vocab_ = nullptr;
    embeddings_enabled_ = false;
}

// ─── Model loading ──────────────────────────────────────────────────────────

bool Engine::load_model(const EngineConfig& config, ProgressCallback progress) {
    cleanup();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = config.n_gpu_layers;
    mparams.use_mmap = config.use_mmap;

    // Set up progress callback via member variable for safe lifetime
    progress_cb_ = progress;
    if (progress_cb_) {
        mparams.progress_callback = [](float p, void* ud) -> bool {
            auto* self = static_cast<Engine*>(ud);
            return self->progress_cb_(p);
        };
        mparams.progress_callback_user_data = this;
    }

    model_.reset(llama_model_load_from_file(config.model_path.c_str(), mparams));
    if (!model_) {
        std::cerr << "[anvil] failed to load model: " << config.model_path << "\n";
        return false;
    }

    vocab_ = llama_model_get_vocab(model_.get());
    if (!vocab_) {
        std::cerr << "[anvil] failed to get vocab\n";
        return false;
    }

    // Pre-allocate token buffer for tokenization
    token_buf_.reserve(4096);
    detok_buf_.resize(4096);

    return true;
}

// ─── MTP assistant loading ──────────────────────────────────────────────────

bool Engine::load_mtp_assistant(const std::string& path) {
    if (!model_) {
        std::cerr << "[anvil] cannot load MTP assistant without a model\n";
        return false;
    }

    auto mparams = llama_model_default_params();
    // MTP assistant inherits target's GPU layers setting if not specified
    mparams.n_gpu_layers = 99;

    mtp_assistant_.reset(llama_model_load_from_file(path.c_str(), mparams));
    if (!mtp_assistant_) {
        std::cerr << "[anvil] failed to load MTP assistant: " << path << "\n";
        return false;
    }

    return true;
}

// ─── Context creation ───────────────────────────────────────────────────────

bool Engine::create_context(const EngineConfig& config) {
    if (!model_) {
        std::cerr << "[anvil] cannot create context without a model\n";
        return false;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = config.n_ctx;
    cparams.n_batch = config.n_batch;
    cparams.n_threads = config.n_threads > 0 ? config.n_threads :
                        static_cast<int32_t>(std::thread::hardware_concurrency());
    cparams.n_threads_batch = config.n_threads_batch > 0 ? config.n_threads_batch :
                             cparams.n_threads;

    // Flash attention
    if (config.flash_attn) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    // Embeddings
    cparams.embeddings = config.embeddings;

    // KV cache types (TurboQuant)
    // Parse cache type strings to ggml_type
    auto parse_cache_type = [](const std::string& s) -> enum ggml_type {
        if (s == "f32")  return GGML_TYPE_F32;
        if (s == "f16")  return GGML_TYPE_F16;
        if (s == "q8_0") return GGML_TYPE_Q8_0;
        if (s == "q4_0") return GGML_TYPE_Q4_0;
        if (s == "q4_1") return GGML_TYPE_Q4_1;
        if (s == "turbo2") return GGML_TYPE_TURBO2_0;
        if (s == "turbo3") return GGML_TYPE_TURBO3_0;
        if (s == "turbo4") return GGML_TYPE_TURBO4_0;
        return GGML_TYPE_F16; // fallback
    };

    cparams.type_k = parse_cache_type(config.cache_type_k);
    cparams.type_v = parse_cache_type(config.cache_type_v);

    ctx_.reset(llama_init_from_model(model_.get(), cparams));
    if (!ctx_) {
        std::cerr << "[anvil] failed to create context\n";
        return false;
    }

    embeddings_enabled_ = config.embeddings;

    // Create sampler chain
    auto sparams = llama_sampler_chain_default_params();
    sampler_.reset(llama_sampler_chain_init(sparams));
    if (!sampler_) {
        std::cerr << "[anvil] failed to create sampler\n";
        return false;
    }

    // Add samplers: top_k -> top_p -> min_p -> temp -> dist
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_top_k(config.top_k));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_top_p(config.top_p, 1));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_min_p(config.min_p, 1));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_temp(config.temperature));
    llama_sampler_chain_add(sampler_.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    return true;
}

// ─── Tokenization ───────────────────────────────────────────────────────────

std::vector<llama_token> Engine::tokenize(const std::string& text, bool add_special, bool parse_special) {
    if (!vocab_) return {};
    return anvil::tokenize(vocab_, text, add_special, parse_special);
}

std::string Engine::detokenize(const llama_token* tokens, int32_t n_tokens) {
    if (!vocab_ || n_tokens <= 0) return {};

    // First call to get required size
    int32_t n = llama_detokenize(vocab_, tokens, n_tokens, nullptr, 0, true, false);
    if (n < 0) return {};

    detok_buf_.resize(n);
    llama_detokenize(vocab_, tokens, n_tokens, detok_buf_.data(), n, true, false);
    return detok_buf_;
}

// ─── Single decode step ─────────────────────────────────────────────────────

int32_t Engine::decode(const llama_batch& batch) {
    return llama_decode(ctx_.get(), batch);
}

// ─── Sampling ───────────────────────────────────────────────────────────────

llama_token Engine::sample() {
    return llama_sampler_sample(sampler_.get(), ctx_.get(), -1);
}

float* Engine::get_logits() {
    return llama_get_logits_ith(ctx_.get(), -1);
}

float* Engine::get_embeddings() {
    if (!embeddings_enabled_) return nullptr;
    return llama_get_embeddings_ith(ctx_.get(), -1);
}

// ─── KV cache management ───────────────────────────────────────────────────

void Engine::kv_clear() {
    if (ctx_) {
        auto mem = llama_get_memory(ctx_.get());
        llama_memory_clear(mem, true);
    }
    if (sampler_) {
        llama_sampler_reset(sampler_.get());
    }
    n_past_ = 0;
}

int32_t Engine::n_past() const {
    return n_past_;
}

// ─── Generation ─────────────────────────────────────────────────────────────

GenerateResult Engine::generate(
    const std::string& prompt,
    const EngineConfig& config,
    std::function<void(const std::string&)> on_token,
    bool apply_template
) {
    GenerateResult result;

    if (!model_ || !ctx_) {
        std::cerr << "[anvil] engine not initialized\n";
        return result;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // Apply chat template if requested and the model has one
    std::string formatted = prompt;
    if (apply_template) {
        const char* tmpl = llama_model_chat_template(model_.get(), nullptr);
        if (tmpl && tmpl[0]) {
            llama_chat_message msg = {"user", prompt.c_str()};
            int32_t n = llama_chat_apply_template(tmpl, &msg, 1, true, nullptr, 0);
            if (n > 0) {
                std::vector<char> buf(n + 1);
                llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), n + 1);
                formatted = std::string(buf.data(), n);
            }
        }
    }

    // Tokenize prompt
    // Note: add_special=false because the chat template (if used) already handles BOS.
    auto tokens = tokenize(formatted, false, true);
    if (tokens.empty()) {
        std::cerr << "[anvil] tokenization failed\n";
        return result;
    }

    // Prompt processing
    {
        int32_t n_tokens = static_cast<int32_t>(tokens.size());
        Batch batch(n_tokens, 0, 1);
        auto* b = batch.get();

        for (int32_t i = 0; i < n_tokens; ++i) {
            b->token[i] = tokens[i];
            b->pos[i] = static_cast<llama_pos>(i);
            b->n_seq_id[i] = 1;
            b->seq_id[i][0] = 0;
            b->logits[i] = (i == n_tokens - 1) ? 1 : 0;
        }

        int32_t ret = decode(*b);
        if (ret < 0) {
            std::cerr << "[anvil] prompt decode failed: " << ret << "\n";
            return result;
        }
    }

    auto t_prompt_end = std::chrono::high_resolution_clock::now();
    result.prompt_eval_ms = std::chrono::duration<float, std::milli>(t_prompt_end - t_start).count();

    // Generation loop
    n_past_ = static_cast<int32_t>(tokens.size());
    // -1 means unlimited — generate until EOS or context limit
    int32_t remaining = static_cast<int32_t>(config.n_ctx) - n_past_;
    int32_t max_gen = config.max_tokens > 0 ? config.max_tokens : std::max(remaining, 1);

    // Pre-allocate single-token batch for reuse (avoids per-token malloc/free)
    Batch gen_batch(1, 0, 1);
    auto* gb = gen_batch.get();
    gb->n_seq_id[0] = 1;
    gb->seq_id[0][0] = 0;
    gb->logits[0] = 1;

    for (int32_t i = 0; i < max_gen; ++i) {
        // Sample
        llama_token id = sample();

        // Check EOS
        if (llama_vocab_is_eog(vocab_, id)) break;

        // Detokenize this token
        int32_t n = llama_token_to_piece(vocab_, id, detok_buf_.data(),
                                          static_cast<int32_t>(detok_buf_.size()), 0, false);
        if (n > 0) {
            std::string piece(detok_buf_.data(), n);
            result.text += piece;
            if (on_token) on_token(piece);
        }

        result.tokens_generated++;

        // Reuse pre-allocated batch
        gb->token[0] = id;
        gb->pos[0] = n_past_;

        int32_t ret = decode(*gb);
        if (ret < 0) {
            std::cerr << "[anvil] decode failed: " << ret << "\n";
            break;
        }

        n_past_++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.eval_ms = std::chrono::duration<float, std::milli>(t_end - t_prompt_end).count();

    if (result.eval_ms > 0) {
        result.tokens_per_sec = result.tokens_generated * 1000.0f / result.eval_ms;
    }

    return result;
}

// ─── Incremental generation (appends to existing KV cache) ──────────────────

GenerateResult Engine::generate_incremental(
    const std::string& prompt,
    const EngineConfig& config,
    std::function<void(const std::string&)> on_token
) {
    GenerateResult result;

    if (!model_ || !ctx_) {
        std::cerr << "[anvil] engine not initialized\n";
        return result;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // Tokenize only the new text (no chat template - caller handles formatting)
    auto tokens = tokenize(prompt, false, true);
    if (tokens.empty()) {
        std::cerr << "[anvil] tokenization failed\n";
        return result;
    }

    // Process new tokens starting from current n_past_
    {
        int32_t n_tokens = static_cast<int32_t>(tokens.size());
        Batch batch(n_tokens, 0, 1);
        auto* b = batch.get();

        for (int32_t i = 0; i < n_tokens; ++i) {
            b->token[i] = tokens[i];
            b->pos[i] = static_cast<llama_pos>(n_past_ + i);
            b->n_seq_id[i] = 1;
            b->seq_id[i][0] = 0;
            b->logits[i] = (i == n_tokens - 1) ? 1 : 0;
        }

        int32_t ret = decode(*b);
        if (ret < 0) {
            std::cerr << "[anvil] incremental decode failed: " << ret << "\n";
            return result;
        }
    }

    n_past_ += static_cast<int32_t>(tokens.size());

    auto t_prompt_end = std::chrono::high_resolution_clock::now();
    result.prompt_eval_ms = std::chrono::duration<float, std::milli>(t_prompt_end - t_start).count();

    // Generation loop
    int32_t remaining = static_cast<int32_t>(config.n_ctx) - n_past_;
    int32_t max_gen = config.max_tokens > 0 ? config.max_tokens : std::max(remaining, 1);

    Batch gen_batch(1, 0, 1);
    auto* gb = gen_batch.get();
    gb->n_seq_id[0] = 1;
    gb->seq_id[0][0] = 0;
    gb->logits[0] = 1;

    for (int32_t i = 0; i < max_gen; ++i) {
        llama_token id = sample();
        if (llama_vocab_is_eog(vocab_, id)) break;

        int32_t n = llama_token_to_piece(vocab_, id, detok_buf_.data(),
                                          static_cast<int32_t>(detok_buf_.size()), 0, false);
        if (n > 0) {
            std::string piece(detok_buf_.data(), n);
            result.text += piece;
            if (on_token) on_token(piece);
        }

        result.tokens_generated++;

        gb->token[0] = id;
        gb->pos[0] = n_past_;

        int32_t ret = decode(*gb);
        if (ret < 0) {
            std::cerr << "[anvil] decode failed: " << ret << "\n";
            break;
        }

        n_past_++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.eval_ms = std::chrono::duration<float, std::milli>(t_end - t_prompt_end).count();

    if (result.eval_ms > 0) {
        result.tokens_per_sec = result.tokens_generated * 1000.0f / result.eval_ms;
    }

    return result;
}

// ─── Model info ─────────────────────────────────────────────────────────────

uint64_t Engine::model_size() const {
    return model_ ? llama_model_size(model_.get()) : 0;
}

uint64_t Engine::model_params() const {
    return model_ ? llama_model_n_params(model_.get()) : 0;
}

int32_t Engine::n_ctx() const {
    return ctx_ ? llama_n_ctx(ctx_.get()) : 0;
}

int32_t Engine::n_vocab() const {
    return vocab_ ? llama_vocab_n_tokens(vocab_) : 0;
}

} // namespace anvil
