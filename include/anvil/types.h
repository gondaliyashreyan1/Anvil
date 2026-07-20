#pragma once

#include <llama.h>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <cassert>

namespace anvil {

// ─── Custom deleters ────────────────────────────────────────────────────────

struct LlamaModelDeleter {
    void operator()(llama_model* p) const noexcept {
        if (p) llama_model_free(p);
    }
};

struct LlamaContextDeleter {
    void operator()(llama_context* p) const noexcept {
        if (p) llama_free(p);
    }
};

struct LlamaSamplerDeleter {
    void operator()(llama_sampler* p) const noexcept {
        if (p) llama_sampler_free(p);
    }
};

struct LlamaAdapterLoraDeleter {
    void operator()(llama_adapter_lora* p) const noexcept {
        if (p) llama_adapter_lora_free(p);
    }
};

// ─── Move-only RAII handles ─────────────────────────────────────────────────

using UniqueModel = std::unique_ptr<llama_model, LlamaModelDeleter>;
using UniqueContext = std::unique_ptr<llama_context, LlamaContextDeleter>;
using UniqueSampler = std::unique_ptr<llama_sampler, LlamaSamplerDeleter>;
using UniqueAdapterLora = std::unique_ptr<llama_adapter_lora, LlamaAdapterLoraDeleter>;

// ─── RAII batch wrapper ─────────────────────────────────────────────────────

class Batch {
public:
    Batch() : batch_{}, owns_{false} {}

    Batch(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
        : batch_{llama_batch_init(n_tokens, embd, n_seq_max)}, owns_(true)
    {
        if (!batch_.token && !batch_.embd) {
            throw std::runtime_error("llama_batch_init failed");
        }
        // Fork may not set n_tokens in returned struct
        batch_.n_tokens = n_tokens;
    }

    ~Batch() {
        if (owns_) {
            llama_batch_free(batch_);
        }
    }

    // Move-only
    Batch(Batch&& o) noexcept : batch_(o.batch_), owns_(o.owns_) {
        o.batch_ = {};
        o.owns_ = false;
    }

    Batch& operator=(Batch&& o) noexcept {
        if (this != &o) {
            if (owns_) llama_batch_free(batch_);
            batch_ = o.batch_;
            owns_ = o.owns_;
            o.batch_ = {};
            o.owns_ = false;
        }
        return *this;
    }

    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;

    llama_batch* get() noexcept { return &batch_; }
    const llama_batch* get() const noexcept { return &batch_; }
    llama_batch* operator->() noexcept { return &batch_; }
    const llama_batch* operator->() const noexcept { return &batch_; }

    void reset() {
        if (owns_) {
            llama_batch_free(batch_);
        }
        batch_ = {};
        owns_ = false;
    }

    // Allocate a new batch
    void init(int32_t n_tokens, int32_t embd, int32_t n_seq_max) {
        reset();
        batch_ = llama_batch_init(n_tokens, embd, n_seq_max);
        owns_ = true;
    }

private:
    llama_batch batch_;
    bool owns_;
};

// ─── Token buffer ───────────────────────────────────────────────────────────

class TokenBuffer {
public:
    TokenBuffer() = default;

    explicit TokenBuffer(size_t capacity) : tokens_(capacity), n_(0) {}

    void reserve(size_t capacity) {
        tokens_.resize(capacity);
        n_ = 0;
    }

    llama_token* data() noexcept { return tokens_.data(); }
    const llama_token* data() const noexcept { return tokens_.data(); }
    size_t capacity() const noexcept { return tokens_.size(); }
    size_t size() const noexcept { return n_; }

    void set_size(size_t n) { n_ = n; }

    llama_token& operator[](size_t i) { return tokens_[i]; }
    const llama_token& operator[](size_t i) const { return tokens_[i]; }

private:
    std::vector<llama_token> tokens_;
    size_t n_ = 0;
};

// ─── String helpers ─────────────────────────────────────────────────────────

inline std::string token_to_piece(const llama_vocab* vocab, llama_token token, bool special = false) {
    int32_t n = llama_token_to_piece(vocab, token, nullptr, 0, 0, special);
    if (n < 0) return {};
    std::string result(n, '\0');
    llama_token_to_piece(vocab, token, result.data(), n, 0, special);
    return result;
}

inline std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text, bool add_special = false, bool parse_special = false) {
    if (!vocab || text.empty()) return {};
    int32_t n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                               nullptr, 0, add_special, parse_special);
    if (n < 0) {
        // n is negative on failure; absolute value is the required size
        int32_t required = -n;
        std::vector<llama_token> tokens(required);
        int32_t actual = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                                        tokens.data(), required, add_special, parse_special);
        if (actual < 0) {
            throw std::runtime_error("tokenize failed (code=" + std::to_string(actual) + ", text_size=" + std::to_string(text.size()) + ")");
        }
        tokens.resize(actual);
        return tokens;
    }
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                   tokens.data(), n, add_special, parse_special);
    return tokens;
}

} // namespace anvil
