#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace anvil {

struct GPUInfo {
    std::string name;
    std::string vendor;        // "Apple", "NVIDIA", "AMD", "Intel"
    int64_t     vram_mb = 0;
    bool        is_discrete = false;
};

struct HardwareProfile {
    std::string os;            // "macos", "linux", "windows"
    std::string arch;          // "x86_64", "aarch64"
    std::string cpu;
    std::string cpu_features;  // "avx2", "avx512", "neon"
    int64_t     ram_gb = 0;
    int32_t     cpu_cores = 0;
    std::vector<GPUInfo> gpus;

    // Derived recommendations
    std::string best_backend;  // "metal", "cuda", "vulkan", "cpu"
    int32_t     recommended_ngl = 0;
    uint32_t    recommended_ctx = 8192;

    // Print detected hardware
    void print() const;
};

// Detect hardware on this machine
HardwareProfile detect_hardware();

// Get recommended n_gpu_layers based on detected hardware
int32_t auto_ngl(const HardwareProfile& hw);

// Get recommended backend name
std::string auto_backend(const HardwareProfile& hw);

} // namespace anvil
