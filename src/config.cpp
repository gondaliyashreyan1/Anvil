#include "anvil/config.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

#ifdef __APPLE__
#include <sys/stat.h>
#include <pwd.h>
#include <unistd.h>
#endif

#ifdef __linux__
#include <sys/stat.h>
#include <pwd.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#endif

namespace anvil {

// ─── Path helpers ───────────────────────────────────────────────────────────

static std::string home_dir() {
#ifdef _WIN32
    char buf[MAX_PATH];
    if (GetHomeDirectoryA(buf, MAX_PATH)) return buf;
    return "C:\\";
#else
    const char* home = getenv("HOME");
    if (home) return home;
    struct passwd* pw = getpwuid(getuid());
    if (pw) return pw->pw_dir;
    return "/tmp";
#endif
}

static bool ensure_dir(const std::string& path) {
#ifdef _WIN32
    return CreateDirectoryA(path.c_str(), nullptr) || GetLastError() == ERROR_ALREADY_EXISTS;
#else
    mkdir(path.c_str(), 0755);
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

std::string config_dir() {
    std::string dir = home_dir() + "/.anvil";
    ensure_dir(dir);
    return dir;
}

std::string config_path() {
    return config_dir() + "/config.json";
}

std::string models_dir() {
    std::string dir = config_dir() + "/models";
    ensure_dir(dir);
    return dir;
}

// ─── Simple JSON parser/writer ──────────────────────────────────────────────
// Minimal JSON for config - no external dependencies

struct JsonValue {
    std::string str_val;
    int64_t int_val = 0;
    double float_val = 0.0;
    bool bool_val = false;
    enum Type { STRING, INT, FLOAT, BOOL, NULL_VAL } type = NULL_VAL;
};

static std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";

    pos = json.find_first_not_of(" \t\n", pos + 1);
    if (pos == std::string::npos) return "";

    if (json[pos] == '"') {
        pos++;
        size_t end = json.find('"', pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    }

    size_t end = json.find_first_of(",}\n", pos);
    if (end == std::string::npos) end = json.size();
    return json.substr(pos, end - pos);
}

static int64_t json_get_int(const std::string& json, const std::string& key, int64_t def = 0) {
    std::string val = json_get_string(json, key);
    if (val.empty()) return def;
    try { return std::stoll(val); }
    catch (...) { return def; }
}

static double json_get_float(const std::string& json, const std::string& key, double def = 0.0) {
    std::string val = json_get_string(json, key);
    if (val.empty()) return def;
    try { return std::stod(val); }
    catch (...) { return def; }
}

static bool json_get_bool(const std::string& json, const std::string& key, bool def = false) {
    std::string val = json_get_string(json, key);
    if (val.empty()) return def;
    return val == "true" || val == "1";
}

// ─── Config I/O ─────────────────────────────────────────────────────────────

EngineConfig load_config() {
    EngineConfig config;

    std::ifstream file(config_path());
    if (!file.is_open()) {
        // Return defaults if no config file
        return config;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());

    // Parse config fields
    config.model_path = json_get_string(content, "model_path");
    config.mtp_head_path = json_get_string(content, "mtp_head_path");
    config.draft_model_path = json_get_string(content, "draft_model_path");

    config.n_gpu_layers = static_cast<int32_t>(json_get_int(content, "n_gpu_layers", -1));
    config.n_ctx = static_cast<int32_t>(json_get_int(content, "n_ctx", 8192));



    config.cache_type_k = json_get_string(content, "cache_type_k");
    if (config.cache_type_k.empty()) config.cache_type_k = "turbo3";
    config.cache_type_v = json_get_string(content, "cache_type_v");
    if (config.cache_type_v.empty()) config.cache_type_v = "turbo3";

    config.flash_attn = json_get_bool(content, "flash_attn");
    config.embeddings = json_get_bool(content, "embeddings");
    config.use_mmap = json_get_bool(content, "use_mmap", true);

    config.spec_type = json_get_string(content, "spec_type");
    if (config.spec_type.empty()) config.spec_type = "none";
    config.draft_block_size = static_cast<int32_t>(json_get_int(content, "draft_block_size", 4));
    config.draft_max = static_cast<int32_t>(json_get_int(content, "draft_max", 16));
    config.draft_min = static_cast<int32_t>(json_get_int(content, "draft_min", 0));

    config.system_prompt = json_get_string(content, "system_prompt");

    return config;
}

bool save_config(const EngineConfig& config) {
    std::ofstream file(config_path());
    if (!file.is_open()) {
        std::cerr << "[anvil] failed to write config: " << config_path() << "\n";
        return false;
    }

    file << "{\n";
    file << "  \"model_path\": \"" << config.model_path << "\",\n";
    file << "  \"mtp_head_path\": \"" << config.mtp_head_path << "\",\n";
    file << "  \"draft_model_path\": \"" << config.draft_model_path << "\",\n";
    file << "  \"n_gpu_layers\": " << config.n_gpu_layers << ",\n";
    file << "  \"n_ctx\": " << config.n_ctx << ",\n";

    file << "  \"cache_type_k\": \"" << config.cache_type_k << "\",\n";
    file << "  \"cache_type_v\": \"" << config.cache_type_v << "\",\n";
    file << "  \"flash_attn\": " << (config.flash_attn ? "true" : "false") << ",\n";
    file << "  \"embeddings\": " << (config.embeddings ? "true" : "false") << ",\n";
    file << "  \"use_mmap\": " << (config.use_mmap ? "true" : "false") << ",\n";
    file << "  \"spec_type\": \"" << config.spec_type << "\",\n";
    file << "  \"draft_block_size\": " << config.draft_block_size << ",\n";
    file << "  \"draft_max\": " << config.draft_max << ",\n";
    file << "  \"draft_min\": " << config.draft_min << ",\n";
    file << "  \"system_prompt\": \"" << config.system_prompt << "\"\n";
    file << "}\n";

    return true;
}

EngineConfig merge_config(const EngineConfig& base, const EngineConfig& cli) {
    EngineConfig merged = base;

    // CLI overrides config file (only if CLI value differs from default)
    if (!cli.model_path.empty()) merged.model_path = cli.model_path;
    if (!cli.mtp_head_path.empty()) merged.mtp_head_path = cli.mtp_head_path;
    if (!cli.draft_model_path.empty()) merged.draft_model_path = cli.draft_model_path;

    if (cli.n_gpu_layers >= 0) merged.n_gpu_layers = cli.n_gpu_layers;
    if (cli.n_ctx > 0) merged.n_ctx = cli.n_ctx;
    if (cli.n_threads > 0) merged.n_threads = cli.n_threads;

    if (cli.temperature >= 0) merged.temperature = cli.temperature;
    if (cli.top_k > 0) merged.top_k = cli.top_k;
    if (cli.top_p > 0) merged.top_p = cli.top_p;
    if (cli.min_p >= 0) merged.min_p = cli.min_p;
    if (cli.max_tokens > 0) merged.max_tokens = cli.max_tokens;

    if (!cli.cache_type_k.empty() && cli.cache_type_k != "turbo3") merged.cache_type_k = cli.cache_type_k;
    if (!cli.cache_type_v.empty() && cli.cache_type_v != "turbo3") merged.cache_type_v = cli.cache_type_v;

    if (cli.flash_attn) merged.flash_attn = cli.flash_attn;
    if (cli.embeddings) merged.embeddings = cli.embeddings;
    if (!cli.use_mmap) merged.use_mmap = cli.use_mmap;

    if (!cli.spec_type.empty() && cli.spec_type != "none") merged.spec_type = cli.spec_type;
    if (cli.draft_block_size != 4) merged.draft_block_size = cli.draft_block_size;
    if (cli.draft_max != 16) merged.draft_max = cli.draft_max;
    if (cli.draft_min != 0) merged.draft_min = cli.draft_min;

    if (!cli.system_prompt.empty()) merged.system_prompt = cli.system_prompt;

    return merged;
}

} // namespace anvil
