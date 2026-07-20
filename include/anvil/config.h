#pragma once

#include "engine.h"
#include <string>

namespace anvil {

// ─── Config file paths ──────────────────────────────────────────────────────

std::string config_dir();
std::string config_path();
std::string models_dir();

// ─── Config I/O ─────────────────────────────────────────────────────────────

// Load config from disk. Returns default config if file doesn't exist.
EngineConfig load_config();

// Save config to disk
bool save_config(const EngineConfig& config);

// Merge CLI flags into config (CLI overrides config file)
EngineConfig merge_config(const EngineConfig& base, const EngineConfig& cli_override);

} // namespace anvil
