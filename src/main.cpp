#include "anvil/engine.h"
#include "anvil/hardware.h"
#include "anvil/config.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <csignal>
#include <filesystem>
#include <iomanip>

// ─── Globals ────────────────────────────────────────────────────────────────

static anvil::Engine g_engine;
static volatile bool g_running = true;

static void signal_handler(int /*sig*/) {
    g_running = false;
}

// ─── Usage ──────────────────────────────────────────────────────────────────

static void print_usage() {
    std::cout << "\nanvil - forge anything\n"
              << "\nUsage:\n"
              << "  anvil [options]                  Run with model picker (interactive)\n"
              << "  anvil run <model> [options]       Run a specific model\n"
              << "  anvil --version                   Show version\n"
              << "  anvil --help                      Show this help\n"
              << "\nRun options:\n"
              << "  -m, --model <path>                Model GGUF path\n"
              << "  --temp, --temperature <f>         Sampling temperature (default: 0.8)\n"
              << "  --max-tokens <n>                  Max tokens to generate (default: unlimited)\n"
              << "  --ctx, --context <n>              Context size (default: 8192)\n"
              << "  --ngl, --gpu-layers <n>           GPU layers to offload (-1 = auto)\n"
              << "  --cache-type-k <type>             K cache type (default: turbo3)\n"
              << "  --cache-type-v <type>             V cache type (default: turbo3)\n"
              << "  --flash-attn                      Enable flash attention\n"
              << "  --system-prompt <text>            System prompt\n"
              << "  --hardware                        Show detected hardware and exit\n"
              << "\nExamples:\n"
              << "  anvil                             # Interactive model picker\n"
              << "  anvil run model.gguf --temp 0.3 --ctx 128000\n"
              << "  anvil run gemma4.gguf --mtp-head assistant.gguf --spec-type mtp\n"
              << std::endl;
}

static void print_version() {
    std::cout << "anvil v0.1.2 (llama-turbo with TurboQuant + MTP + NextN)" << std::endl;
}

// ─── Model discovery ────────────────────────────────────────────────────────

static std::vector<std::string> discover_models() {
    std::vector<std::string> models;
    std::string models_dir = anvil::models_dir();

    if (!std::filesystem::exists(models_dir)) {
        return models;
    }

    for (const auto& entry : std::filesystem::directory_iterator(models_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            models.push_back(entry.path().filename().string());
        }
    }

    std::sort(models.begin(), models.end());
    return models;
}

static std::string interactive_model_picker() {
    auto models = discover_models();

    if (models.empty()) {
        std::cerr << "\n[anvil] No GGUF models found in " << anvil::models_dir() << "\n\n";
        std::cerr << "  Download a model to get started:\n\n";
        std::cerr << "    # Qwen 3.5 0.8B (fast, 542 MB)\n";
        std::cerr << "    huggingface-cli download bartowski/Qwen_Qwen3.5-0.8B-GGUF \\\n";
        std::cerr << "      --include '*Q4_K_M.gguf' \\\n";
        std::cerr << "      --local-dir " << anvil::models_dir() << "\n\n";
        std::cerr << "    # Or any GGUF from HuggingFace\n";
        std::cerr << "    huggingface-cli download <repo> \\\n";
        std::cerr << "      --include '*.gguf' \\\n";
        std::cerr << "      --local-dir " << anvil::models_dir() << "\n\n";
        return "";
    }

    if (models.size() == 1) {
        std::cout << "[anvil] Found 1 model: " << models[0] << "\n";
        return anvil::models_dir() + "/" + models[0];
    }

    std::cout << "\n[anvil] Available models:\n\n";
    for (size_t i = 0; i < models.size(); ++i) {
        std::error_code ec;
        auto size = std::filesystem::file_size(anvil::models_dir() + "/" + models[i], ec);
        if (ec) continue;
        double mb = size / (1024.0 * 1024.0);
        std::cout << "  " << (i + 1) << ") " << models[i];
        if (mb >= 1024.0) {
            std::cout << "  (" << std::fixed << std::setprecision(1) << (mb / 1024.0) << " GB)";
        } else {
            std::cout << "  (" << std::fixed << std::setprecision(0) << mb << " MB)";
        }
        std::cout << "\n";
    }
    std::cout << "\n  Select model [1-" << models.size() << "]: ";
    std::cout << std::flush;

    std::string input;
    if (!std::getline(std::cin, input)) return "";

    size_t choice = 0;
    try {
        choice = std::stoul(input);
    } catch (...) {
        std::cerr << "[anvil] invalid selection\n";
        return "";
    }

    if (choice < 1 || choice > models.size()) {
        std::cerr << "[anvil] invalid selection (1-" << models.size() << ")\n";
        return "";
    }

    return anvil::models_dir() + "/" + models[choice - 1];
}

// ─── Argument parsing ───────────────────────────────────────────────────────

struct CliArgs {
    std::string command;          // "run", "serve", ""
    std::string model_path;
    anvil::EngineConfig config;
    bool show_help = false;
    bool show_version = false;
    bool show_hardware = false;
    bool no_tui = false;
};

static CliArgs parse_args(int argc, char** argv) {
    CliArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            args.show_help = true;
        } else if (arg == "--version" || arg == "-v") {
            args.show_version = true;
        } else if (arg == "--hardware") {
            args.show_hardware = true;
        } else if (arg == "--no-tui") {
            args.no_tui = true;
        } else if (arg == "--flash-attn") {
            args.config.flash_attn = true;
        } else if (arg == "run" || arg == "serve") {
            args.command = arg;
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            args.config.model_path = argv[++i];
        } else if ((arg == "--temp" || arg == "--temperature") && i + 1 < argc) {
            args.config.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            args.config.top_k = std::stoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            args.config.top_p = std::stof(argv[++i]);
        } else if (arg == "--min-p" && i + 1 < argc) {
            args.config.min_p = std::stof(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            args.config.max_tokens = std::stoi(argv[++i]);
        } else if ((arg == "--ctx" || arg == "--context") && i + 1 < argc) {
            args.config.n_ctx = std::stoi(argv[++i]);
        } else if ((arg == "--ngl" || arg == "--gpu-layers") && i + 1 < argc) {
            args.config.n_gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--backend" && i + 1 < argc) {
            std::string backend = argv[++i];
            if (backend == "metal") args.config.n_gpu_layers = 99;
            else if (backend == "cuda") args.config.n_gpu_layers = 99;
            else if (backend == "cpu") args.config.n_gpu_layers = 0;
        } else if (arg == "--cache-type-k" && i + 1 < argc) {
            args.config.cache_type_k = argv[++i];
        } else if (arg == "--cache-type-v" && i + 1 < argc) {
            args.config.cache_type_v = argv[++i];
        } else if (arg == "--spec-type" && i + 1 < argc) {
            args.config.spec_type = argv[++i];
        } else if (arg == "--mtp-head" && i + 1 < argc) {
            args.config.mtp_head_path = argv[++i];
        } else if (arg == "--draft-block-size" && i + 1 < argc) {
            args.config.draft_block_size = std::stoi(argv[++i]);
        } else if (arg == "--draft-max" && i + 1 < argc) {
            args.config.draft_max = std::stoi(argv[++i]);
        } else if (arg == "--system-prompt" && i + 1 < argc) {
            args.config.system_prompt = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            ++i;
        } else if (arg == "--port" && i + 1 < argc) {
            ++i;
        } else if (arg[0] != '-') {
            if (args.config.model_path.empty()) {
                args.config.model_path = arg;
            }
        }
    }

    return args;
}

// ─── REPL mode ──────────────────────────────────────────────────────────────

static void run_repl(anvil::Engine& engine, const anvil::EngineConfig& config) {
    std::cout << "[anvil] REPL mode. Type your prompt and press Enter.\n";
    std::cout << "[anvil] Type '/quit' or '/exit' to stop.\n\n";

    std::string input;
    while (g_running) {
        std::cout << "> ";
        if (!std::getline(std::cin, input)) break;

        if (input == "/quit" || input == "/exit") break;
        if (input.empty()) continue;

        // Clear KV cache between REPL turns
        engine.kv_clear();

        auto result = engine.generate(input, config, [](const std::string& token) {
            std::cout << token << std::flush;
        });
        std::cout << "\n\n";

        std::cout << "[stats] " << result.tokens_generated << " tokens"
                  << " | " << result.prompt_eval_ms << "ms prompt"
                  << " | " << result.eval_ms << "ms gen"
                  << " | " << result.tokens_per_sec << " tok/s\n\n";
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto args = parse_args(argc, argv);

    if (args.show_help) { print_usage(); return 0; }
    if (args.show_version) { print_version(); return 0; }

    if (args.show_hardware) {
        auto hw = anvil::detect_hardware();
        hw.print();
        return 0;
    }

    anvil::Engine::backend_init();

    auto file_config = anvil::load_config();
    auto config = anvil::merge_config(file_config, args.config);

    // Interactive model selection if no model specified
    if (config.model_path.empty()) {
        config.model_path = interactive_model_picker();
        if (config.model_path.empty()) {
            anvil::Engine::backend_free();
            return 1;
        }
    }

    if (config.n_gpu_layers < 0) {
        auto hw = anvil::detect_hardware();
        hw.print();
        config.n_gpu_layers = anvil::auto_ngl(hw);
        if (config.flash_attn && hw.best_backend == "metal") {
            config.flash_attn = true;
        }
    }

    std::cout << "[anvil] Loading model: " << config.model_path << "\n";
    std::cout << "[anvil] KV cache: K=" << config.cache_type_k << " V=" << config.cache_type_v << "\n";
    std::cout << "[anvil] GPU layers: " << config.n_gpu_layers << "\n";
    std::cout << "[anvil] Context: " << config.n_ctx << "\n";

    auto progress_cb = [](float p) -> bool {
        std::cout << "\r[anvil] Loading... " << static_cast<int>(p * 100) << "%" << std::flush;
        return g_running;
    };

    if (!g_engine.load_model(config, progress_cb)) {
        std::cerr << "\n[anvil] Failed to load model\n";
        anvil::Engine::backend_free();
        return 1;
    }
    std::cout << "\n";

    if (!config.mtp_head_path.empty()) {
        std::cout << "[anvil] Loading MTP assistant: " << config.mtp_head_path << "\n";
        if (!g_engine.load_mtp_assistant(config.mtp_head_path)) {
            std::cerr << "[anvil] Failed to load MTP assistant\n";
            anvil::Engine::backend_free();
            return 1;
        }
    }

    if (!g_engine.create_context(config)) {
        std::cerr << "[anvil] Failed to create context\n";
        anvil::Engine::backend_free();
        return 1;
    }

    anvil::save_config(config);

    std::cout << "[anvil] Model loaded: "
              << (g_engine.model_size() / (1024 * 1024)) << " MB"
              << " | " << g_engine.model_params() << " params"
              << " | " << g_engine.n_ctx() << " ctx"
              << " | " << g_engine.n_vocab() << " vocab\n";

    if (args.command == "serve") {
        std::cout << "[anvil] API server not yet implemented. Use REPL mode.\n";
        run_repl(g_engine, config);
    } else {
        run_repl(g_engine, config);
    }

    // CRITICAL: Cleanup engine resources BEFORE backend_free to avoid Metal crash
    g_engine.cleanup();
    anvil::Engine::backend_free();
    return 0;
}
