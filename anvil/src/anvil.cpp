#include "llama.h"
#include "ggml.h"
#include <algorithm>
#include <atomic>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#ifdef __APPLE__
#include <sys/sysctl.h>
#include <IOKit/IOKitLib.h>
#endif
#ifdef __linux__
#include <glob.h>
#include <fstream>
#include <regex>
#endif
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi.h>
#pragma comment(lib, "dxgi.lib")
#endif
static const char * ANVIL_LOGO = R"(
   ░███                          ░██░██ 
  ░██░██                            ░██ 
 ░██  ░██  ░████████  ░██    ░██ ░██░██ 
░█████████ ░██    ░██ ░██    ░██ ░██░██ 
░██    ░██ ░██    ░██  ░██  ░██  ░██░██ 
░██    ░██ ░██    ░██   ░██░██   ░██░██ 
░██    ░██ ░██    ░██    ░███    ░██░██
)";
static std::atomic<bool> g_interrupted{false};
static void signal_handler(int) {
    if (g_interrupted.load()) {
        fprintf(stdout, "\033[0m\n");
        fflush(stdout);
        std::exit(130);
    }
    g_interrupted.store(true);
}
struct GPUInfo {
    std::string name;
    std::string vendor;  
    uint64_t vram_mb = 0;
    bool is_discrete = false;
};
struct AnvilConfig {
    int  ngl         = 99;
    int  n_ctx       = 8192;
    float temp       = 0.8f;
    bool flash_attn  = true;
    bool mtp         = false;
    bool no_turbo    = false;
    std::string model;
};
static std::string config_dir() {
    const char * home = getenv("HOME");
    if (!home) home = ".";
    return std::string(home) + "/.anvil";
}
static std::string config_path() {
    return config_dir() + "/config.json";
}
static void write_config(const AnvilConfig & cfg) {
    namespace fs = std::filesystem;
    fs::create_directories(config_dir());
    std::ofstream f(config_path());
    if (!f) return;
    f << "{\n";
    f << "  \"ngl\": " << cfg.ngl << ",\n";
    f << "  \"n_ctx\": " << cfg.n_ctx << ",\n";
    f << "  \"temp\": " << cfg.temp << ",\n";
    f << "  \"flash_attn\": " << (cfg.flash_attn ? "true" : "false") << ",\n";
    f << "  \"mtp\": " << (cfg.mtp ? "true" : "false") << ",\n";
    f << "  \"no_turbo\": " << (cfg.no_turbo ? "true" : "false") << ",\n";
    f << "  \"model\": \"" << cfg.model << "\"\n";
    f << "}\n";
}
static std::string json_get(const std::string & json, const std::string & key) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos >= json.size()) return "";
    if (json[pos] == '"') {
        pos++;
        auto end = json.find('"', pos);
        return json.substr(pos, end - pos);
    }
    auto end = pos;
    while (end < json.size() && json[end] != ',' && json[end] != '}' && json[end] != '\n') end++;
    return json.substr(pos, end - pos);
}
static AnvilConfig load_config() {
    AnvilConfig cfg;
    std::ifstream f(config_path());
    if (!f) return cfg;
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto s = json_get(json, "ngl");
    if (!s.empty()) cfg.ngl = std::stoi(s);
    s = json_get(json, "n_ctx");
    if (!s.empty()) cfg.n_ctx = std::stoi(s);
    s = json_get(json, "temp");
    if (!s.empty()) cfg.temp = std::stof(s);
    s = json_get(json, "flash_attn");
    if (!s.empty()) cfg.flash_attn = (s == "true");
    s = json_get(json, "mtp");
    if (!s.empty()) cfg.mtp = (s == "true");
    s = json_get(json, "no_turbo");
    if (!s.empty()) cfg.no_turbo = (s == "true");
    s = json_get(json, "model");
    if (!s.empty()) cfg.model = s;
    return cfg;
}
struct HWInfo {
    std::string os;
    std::string arch;
    std::string cpu;
    uint64_t ram_bytes = 0;
    std::vector<GPUInfo> gpus;
    bool apple_silicon = false;
};
#ifdef __APPLE__
static void detect_gpus_macos(HWInfo & hw) {
    CFMutableDictionaryRef matching = IOServiceMatching("IOGPU");
    io_iterator_t iter;
    kern_return_t kr = IOServiceGetMatchingServices(kIOMasterPortDefault, matching, &iter);
    if (kr != KERN_SUCCESS) return;

    io_object_t device;
    while ((device = IOIteratorNext(iter)) != 0) {
        GPUInfo gpu;
        gpu.vendor = "Apple";
        gpu.is_discrete = false;

        CFTypeRef name_ref = IORegistryEntrySearchCFProperty(
            device, kIOServicePlane, CFSTR("gpu-id"),
            kCFAllocatorDefault, kIORegistryIterateRecursively);
        uint64_t mem = 0;
        size_t mem_len = sizeof(mem);
        sysctlbyname("hw.memsize", &mem, &mem_len, nullptr, 0);
        gpu.vram_mb = mem / (1024 * 1024);
        gpu.name = hw.cpu + " GPU";
        hw.gpus.push_back(gpu);
        IOObjectRelease(device);
    }
    IOObjectRelease(iter);
}
#endif
#ifdef __linux__
static void run_cmd(const char * cmd, std::string & out) {
    FILE * pipe = popen(cmd, "r");
    if (!pipe) return;
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);
}
static void detect_gpus_linux(HWInfo & hw) {
    {
        std::string out;
        run_cmd("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null", out);
        if (!out.empty()) {
            std::istringstream ss(out);
            std::string line;
            while (std::getline(ss, line)) {
                while (!line.empty() && line.back() <= ' ') line.pop_back();
                if (line.empty()) continue;
                GPUInfo gpu;
                gpu.vendor = "NVIDIA";
                gpu.is_discrete = true;
                auto comma = line.rfind(',');
                if (comma != std::string::npos) {
                    gpu.name = line.substr(0, comma);
                    while (!gpu.name.empty() && gpu.name.back() == ' ') gpu.name.pop_back();
                    gpu.vram_mb = std::stoull(line.substr(comma + 1));
                } else {
                    gpu.name = line;
                }
                hw.gpus.push_back(gpu);
            }
        }
    }
    {
        glob_t globbuf;
        if (glob("/sys/class/drm/card*/device/vendor", 0, nullptr, &globbuf) == 0) {
            for (size_t i = 0; i < globbuf.gl_pathc; i++) {
                std::ifstream vf(globbuf.gl_pathv[i]);
                std::string vendor_id;
                std::getline(vf, vendor_id);
                while (!vendor_id.empty() && vendor_id.back() <= ' ') vendor_id.pop_back();
                std::string device_path = std::string(globbuf.gl_pathv[i]).substr(0,
                    std::string(globbuf.gl_pathv[i]).rfind("/device/vendor"));
                device_path += "/device";
                std::ifstream df(device_path);
                std::string device_id;
                std::getline(df, device_id);
                while (!device_id.empty() && device_id.back() <= ' ') device_id.pop_back();
                GPUInfo gpu;
                if (vendor_id == "0x1002") {
                    gpu.vendor = "AMD";
                    gpu.is_discrete = true;
                    gpu.name = "AMD GPU (" + device_id + ")";
                } else if (vendor_id == "0x8086") {
                    gpu.vendor = "Intel";
                    gpu.is_discrete = false;
                    gpu.name = "Intel GPU (" + device_id + ")";
                } else {
                    globfree(&globbuf);
                    continue;
                }
                std::string vendor_path(globbuf.gl_pathv[i]);
                std::string card_dir = vendor_path.substr(0, vendor_path.rfind("/device"));
                std::string vram_path = card_dir + "/device/mem_info_vram_total";
                std::ifstream vf2(vram_path);
                if (vf2.good()) {
                    uint64_t bytes = 0;
                    vf2 >> bytes;
                    gpu.vram_mb = bytes / (1024 * 1024);
                }
                hw.gpus.push_back(gpu);
            }
            globfree(&globbuf);
        }
    }
}
#endif
#ifdef _WIN32
static void detect_gpus_windows(HWInfo & hw) {
    IDXGIFactory * factory = nullptr;
    if (CreateDXGIFactory(__uuidof(IDXGIFactory), (void **)&factory) != S_OK) return;
    IDXGIAdapter * adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC desc;
        if (adapter->GetDesc(&desc) == S_OK) {
            GPUInfo gpu;
            char name_buf[256];
            wcstombs(name_buf, desc.Description, sizeof(name_buf));
            gpu.name = name_buf;
            gpu.vram_mb = desc.DedicatedVideoMemory / (1024 * 1024);
            gpu.is_discrete = (desc.VendorId == 0x10DE); // NVIDIA
            if (desc.VendorId == 0x10DE) gpu.vendor = "NVIDIA";
            else if (desc.VendorId == 0x1002) gpu.vendor = "AMD";
            else if (desc.VendorId == 0x8086) gpu.vendor = "Intel";
            else gpu.vendor = "Other";
            hw.gpus.push_back(gpu);
        }
        adapter->Release();
    }
    factory->Release();
}
#endif
static HWInfo probe_hw() {
    HWInfo hw;
#if defined(__APPLE__)
    hw.os = "macos";
#elif defined(__linux__)
    hw.os = "linux";
#elif defined(_WIN32)
    hw.os = "windows";
#else
    hw.os = "unknown";
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    hw.arch = "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
    hw.arch = "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    hw.arch = "i386";
#else
    hw.arch = "unknown";
#endif
#ifdef __APPLE__
    {
        char buf[256];
        size_t len = sizeof(buf);
        if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) {
            hw.cpu = buf;
            hw.apple_silicon = (hw.cpu.find("Apple") != std::string::npos);
        }
    }
    {
        uint64_t ram = 0;
        size_t len = sizeof(ram);
        sysctlbyname("hw.memsize", &ram, &len, nullptr, 0);
        hw.ram_bytes = ram;
    }
    detect_gpus_macos(hw);
#elif defined(__linux__)
    {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("model name") != std::string::npos) {
                auto pos = line.find(':');
                if (pos != std::string::npos) {
                    hw.cpu = line.substr(pos + 2);
                    while (!hw.cpu.empty() && hw.cpu.back() <= ' ') hw.cpu.pop_back();
                }
                break;
            }
        }
    }
    {
        std::ifstream f("/proc/meminfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("MemTotal") != std::string::npos) {
                uint64_t kb = 0;
                sscanf(line.c_str(), "MemTotal: %lu kB", &kb);
                hw.ram_bytes = kb * 1024;
                break;
            }
        }
    }
    detect_gpus_linux(hw);
#elif defined(_WIN32)
    {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        hw.arch = (si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_ARM64) ? "aarch64" : "x86_64";
        hw.cpu = "Unknown CPU";
    }
    {
        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) {
            hw.ram_bytes = ms.ullTotalPhys;
        }
    }
    detect_gpus_windows(hw);
#endif

    return hw;
}
static int derive_ngl(const HWInfo & hw) {
    if (hw.apple_silicon) return 99;
    for (const auto & gpu : hw.gpus) {
        if (gpu.is_discrete && gpu.vram_mb >= 4096) return 99;
    }
    for (const auto & gpu : hw.gpus) {
        if (gpu.vram_mb >= 4096) return 99;
    }
    return 0; 
}
static int derive_ctx(uint64_t ram_bytes) {
    uint64_t gb = ram_bytes / (1024ULL * 1024 * 1024);
    if (gb >= 48) return 131072;
    if (gb >= 24) return 65536;
    if (gb >= 12) return 32768;
    return 8192;
}
struct CliArgs {
    std::string model;
    int  n_ctx       = 0;
    int  ngl         = -1;
    float temp       = -1;
    bool flash_attn  = false;
    bool no_flash_attn = false;
    bool mtp         = false;
    bool no_turbo    = false;
    bool help        = false;
    bool version     = false;
    std::string system_prompt;
    std::string prompt;
    int  max_tokens  = 4096;
};
static void print_usage() {
    printf("anvil — Forge anything.\n\n");
    printf("Usage:\n");
    printf("  anvil run <model> [options]     Run a model with chat REPL\n");
    printf("  anvil run <model> -p \"prompt\"   Single-shot generation\n");
    printf("  anvil --help                    Show this help\n");
    printf("  anvil --version                 Show version\n\n");
    printf("Options:\n");
    printf("  -c, --ctx <n>            Context size (default: auto from RAM)\n");
    printf("  -ngl, --ngl, --n-gpu-layers <n> GPU layers to offload (default: auto)\n");
    printf("  -t, --temp <f>           Sampling temperature (default: 0.8)\n");
    printf("  --flash-attn             Enable flash attention (default: on)\n");
    printf("  --no-flash-attn          Disable flash attention\n");
    printf("  --mtp                    Enable MTP speculative decoding (Gemma 4)\n");
    printf("  --no-turbo               Disable TurboQuant KV cache compression\n");
    printf("  -s, --system <text>      System prompt\n");
    printf("  -p, --prompt <text>      User prompt (non-interactive mode)\n");
    printf("  -n, --max-tokens <n>     Max tokens to generate (default: 4096)\n\n");
    printf("TurboQuant KV: K=turbo4 (~3.8x) V=turbo3 (~4.9x) by default.\n");
    printf("Override via --no-turbo or set in ~/.anvil/config.json\n\n");
    printf("Examples:\n");
    printf("  anvil run llama3.1\n");
    printf("  anvil run model.gguf --ctx 131072 --ngl 99\n");
    printf("  anvil run model.gguf -p \"Explain quantum computing\" -n 200\n");
}

static CliArgs parse_args(int argc, char ** argv) {
    CliArgs a;
    if (argc < 2) { a.help = true; return a; }

    int i = 1;
    if (i < argc && std::string(argv[i]) == "run") i++;

    for (; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { a.help = true; }
        else if (arg == "--version") { a.version = true; }
        else if ((arg == "-c" || arg == "--ctx") && i + 1 < argc) { a.n_ctx = std::stoi(argv[++i]); }
        else if ((arg == "-ngl" || arg == "--ngl" || arg == "--n-gpu-layers") && i + 1 < argc) { a.ngl = std::stoi(argv[++i]); }
        else if ((arg == "-t" || arg == "--temp") && i + 1 < argc) { a.temp = std::stof(argv[++i]); }
        else if (arg == "--flash-attn") { a.flash_attn = true; }
        else if (arg == "--no-flash-attn") { a.no_flash_attn = true; a.flash_attn = false; }
        else if (arg == "--mtp") { a.mtp = true; }
        else if (arg == "--no-turbo") { a.no_turbo = true; }
        else if ((arg == "-s" || arg == "--system") && i + 1 < argc) { a.system_prompt = argv[++i]; }
        else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) { a.prompt = argv[++i]; }
        else if ((arg == "-n" || arg == "--max-tokens") && i + 1 < argc) { a.max_tokens = std::stoi(argv[++i]); }
        else if (arg[0] != '-') { a.model = arg; }
        else { fprintf(stderr, "Unknown option: %s\n", arg.c_str()); a.help = true; }
    }
    return a;
}

// ---------------------------------------------------------------------------
// GGUF validation
// ---------------------------------------------------------------------------
static bool validate_gguf(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    bool ok = (fread(magic, 1, 4, f) == 4 && memcmp(magic, "GGUF", 4) == 0);
    fclose(f);
    return ok;
}

// ---------------------------------------------------------------------------
// Token helpers
// ---------------------------------------------------------------------------
static std::string token_to_str(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf) - 1, 0, true);
    if (n < 0) return "";
    buf[n] = '\0';
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Chat REPL
// ---------------------------------------------------------------------------
static int run_chat(const CliArgs & cli, const AnvilConfig & cfg) {
    // Load model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.ngl;
    mparams.use_mmap = true;

    fprintf(stderr, "Loading model: %s ...\n", cli.model.c_str());
    llama_model * model = llama_model_load_from_file(cli.model.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "error: failed to load model '%s'\n", cli.model.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Context params — use the fork's real enum values
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = cfg.n_ctx;
    cparams.n_batch = cfg.n_ctx;
    cparams.flash_attn_type = cfg.flash_attn
        ? LLAMA_FLASH_ATTN_TYPE_ENABLED
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k = cfg.no_turbo ? GGML_TYPE_F16 : GGML_TYPE_TURBO4_0;
    cparams.type_v = cfg.no_turbo ? GGML_TYPE_F16 : GGML_TYPE_TURBO3_0;
    if (cfg.mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Sampler chain
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(cfg.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Banner
    printf("\033[1;33m%s\033[0m", ANVIL_LOGO);
    printf("  model  : %s\n", cli.model.c_str());
    printf("  backend: GPU layers=%d | flash=%s\n", cfg.ngl, cfg.flash_attn ? "on" : "off");
    printf("  ctx    : %d\n", cfg.n_ctx);
    printf("  KV     : K=%s V=%s\n",
        cfg.no_turbo ? "f16" : "turbo4_0",
        cfg.no_turbo ? "f16" : "turbo3_0");
    printf("  temp   : %.2f\n", cfg.temp);
    if (cfg.mtp) printf("  spec   : MTP (Gemma 4)\n");
    printf("  type   : /exit to quit, /clear to reset\n\n");

    // Chat state
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(cparams.n_ctx * 4);
    int prev_len = 0;

    if (!cli.system_prompt.empty()) {
        messages.push_back({"system", strdup(cli.system_prompt.c_str())});
    }

    const char * tmpl = llama_model_chat_template(model, nullptr);

    auto generate = [&](const std::string & prompt_text) -> std::string {
        std::string response;
        const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;

        int n_tokens = -llama_tokenize(vocab, prompt_text.c_str(), prompt_text.size(), nullptr, 0, is_first, true);
        if (n_tokens < 0) { fprintf(stderr, "tokenization error\n"); return ""; }
        std::vector<llama_token> tokens(n_tokens);
        llama_tokenize(vocab, prompt_text.c_str(), prompt_text.size(), tokens.data(), tokens.size(), is_first, true);

        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        while (true) {
            int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            if (n_ctx_used + batch.n_tokens > (int)cparams.n_ctx) {
                fprintf(stderr, "\n\033[0mcontext exceeded\033[0m\n");
                return response;
            }
            if (g_interrupted.load()) return response;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode error\n"); return response; }

            llama_token id = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) break;

            std::string piece = token_to_str(vocab, id);
            printf("%s", piece.c_str());
            fflush(stdout);
            response += piece;

            llama_sampler_accept(smpl, id);
            batch = llama_batch_get_one(&id, 1);
        }
        return response;
    };

    if (cli.prompt.empty()) {
        // Interactive mode
        while (true) {
            g_interrupted.store(false);
            printf("\033[32m> \033[0m");
            fflush(stdout);

            std::string user_input;
            if (!std::getline(std::cin, user_input)) break;
            while (!user_input.empty() && user_input.back() == '\n') user_input.pop_back();
            if (user_input.empty()) continue;
            if (user_input == "/exit" || user_input == "/quit") break;
            if (user_input == "/clear") {
                for (auto & msg : messages) free(const_cast<char *>(msg.content));
                messages.clear();
                prev_len = 0;
                llama_memory_clear(llama_get_memory(ctx), true);
                if (!cli.system_prompt.empty()) {
                    messages.push_back({"system", strdup(cli.system_prompt.c_str())});
                }
                printf("Chat cleared.\n\n");
                continue;
            }

            messages.push_back({"user", strdup(user_input.c_str())});

            int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
            if (new_len > (int)formatted.size()) {
                formatted.resize(new_len + 256);
                new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
            }
            if (new_len < 0) {
                fprintf(stderr, "chat template failed, using raw prompt\n");
                std::string raw = user_input + "\n";
                messages.pop_back();
                printf("\033[33m");
                std::string resp = generate(raw);
                printf("\n\033[0m");
                continue;
            }

            std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);
            printf("\033[33m");
            std::string resp = generate(prompt);
            printf("\n\033[0m");
            messages.push_back({"assistant", strdup(resp.c_str())});
            prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
        }
    } else {
        // Single-shot mode
        messages.push_back({"user", strdup(cli.prompt.c_str())});
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len + 256);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        }
        std::string prompt_text;
        if (new_len > 0) {
            prompt_text.assign(formatted.begin(), formatted.begin() + new_len);
        } else {
            prompt_text = cli.prompt;
        }
        printf("\033[33m");
        generate(prompt_text);
        printf("\n\033[0m");
    }

    // Cleanup
    for (auto & msg : messages) free(const_cast<char *>(msg.content));
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    printf("\nExiting.\n");
    return 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) fprintf(stderr, "%s", text);
    }, nullptr);

    CliArgs cli = parse_args(argc, argv);
    if (cli.help) { print_usage(); return 0; }
    if (cli.version) { printf("anvil 0.1.0\n"); return 0; }
    if (cli.model.empty()) { fprintf(stderr, "error: no model specified\n\n"); print_usage(); return 1; }

    AnvilConfig cfg = load_config();
    HWInfo hw = probe_hw();

    // Print hardware info
    fprintf(stderr, "Hardware: %s | %s | %lu GB RAM\n",
        hw.cpu.c_str(), hw.arch.c_str(),
        (unsigned long)(hw.ram_bytes / (1024ULL * 1024 * 1024)));
    for (const auto & gpu : hw.gpus) {
        fprintf(stderr, "GPU: %s (%s) %lu MB VRAM%s\n",
            gpu.name.c_str(), gpu.vendor.c_str(),
            (unsigned long)gpu.vram_mb,
            gpu.is_discrete ? " [discrete]" : "");
    }

    // Resolution order: CLI > config > HW probe > hardcoded
    if (cli.n_ctx > 0)         cfg.n_ctx = cli.n_ctx;
    else if (cfg.n_ctx == 0)   cfg.n_ctx = derive_ctx(hw.ram_bytes);

    if (cli.ngl >= 0)          cfg.ngl = cli.ngl;
    else if (cfg.ngl == 0)     cfg.ngl = derive_ngl(hw);

    if (cli.temp >= 0)         cfg.temp = cli.temp;
    if (cli.flash_attn)        cfg.flash_attn = true;
    if (cli.no_flash_attn)     cfg.flash_attn = false;
    if (cli.mtp)               cfg.mtp = true;
    if (cli.no_turbo)          cfg.no_turbo = true;

    cfg.model = cli.model;
    write_config(cfg);

    // Validate GGUF before loading
    if (!validate_gguf(cli.model)) {
        fprintf(stderr, "error: '%s' is not a valid GGUF file\n", cli.model.c_str());
        return 1;
    }

    llama_backend_init();
    int rc = run_chat(cli, cfg);
    llama_backend_free();
    return rc;
}
