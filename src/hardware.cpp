#include "anvil/hardware.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <cstdlib>
#include <cstring>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

#ifdef __linux__
#include <unistd.h>
#endif

namespace anvil {

// ─── Helpers ────────────────────────────────────────────────────────────────

static std::string read_sysctl(const char* name) {
#ifdef __APPLE__
    char buf[256];
    size_t size = sizeof(buf);
    if (sysctlbyname(name, buf, &size, nullptr, 0) == 0) {
        return std::string(buf);
    }
#endif
    return "";
}

static int64_t read_sysctl_int(const char* name) {
#ifdef __APPLE__
    int64_t val = 0;
    size_t size = sizeof(val);
    if (sysctlbyname(name, &val, &size, nullptr, 0) == 0) {
        return val;
    }
#endif
    return 0;
}

static std::string run_command(const char* cmd) {
    std::string result;
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return result;
    char buf[1024];
    while (fgets(buf, sizeof(buf), pipe)) {
        result += buf;
    }
    pclose(pipe);
    return result;
}

// ─── macOS detection ────────────────────────────────────────────────────────

#ifdef __APPLE__
static void detect_macos(HardwareProfile& hw) {
    hw.os = "macos";

    // Architecture
    hw.arch = read_sysctl("hw.machine");

    // CPU
    hw.cpu = read_sysctl("machdep.cpu.brand_string");
    hw.cpu_cores = static_cast<int32_t>(read_sysctl_int("hw.ncpu"));

    // Check for ARM64
    int64_t is_arm64 = 0;
    size_t size = sizeof(is_arm64);
    sysctlbyname("hw.optional.arm64", &is_arm64, &size, nullptr, 0);
    hw.cpu_features = is_arm64 ? "neon" : "avx2";

    // RAM
    int64_t ram_bytes = read_sysctl_int("hw.memsize");
    hw.ram_gb = ram_bytes / (1024 * 1024 * 1024);

    // GPU (Metal is always available on macOS)
    GPUInfo gpu;
    gpu.vendor = "Apple";
    gpu.is_discrete = false;

    // Try to get GPU name from system_profiler
    std::string sp_output = run_command("system_profiler SPDisplaysDataType 2>/dev/null");
    size_t chip_pos = sp_output.find("Chipset Model:");
    if (chip_pos != std::string::npos) {
        size_t eol = sp_output.find('\n', chip_pos);
        if (eol != std::string::npos) {
            std::string line = sp_output.substr(chip_pos + 15, eol - chip_pos - 15);
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t");
            if (start != std::string::npos) {
                gpu.name = line.substr(start);
            }
        }
    }
    if (gpu.name.empty()) {
        gpu.name = "Apple GPU";
    }

    // Check if discrete GPU
    if (gpu.name.find("AMD") != std::string::npos ||
        gpu.name.find("Radeon") != std::string::npos ||
        gpu.name.find("NVIDIA") != std::string::npos) {
        gpu.is_discrete = true;
        gpu.vendor = gpu.name.find("AMD") != std::string::npos ||
                     gpu.name.find("Radeon") != std::string::npos ? "AMD" : "NVIDIA";
    }

    // Apple Silicon unified memory
    if (is_arm64) {
        gpu.vram_mb = ram_bytes / (1024 * 1024); // unified memory
    }

    hw.gpus.push_back(gpu);
    hw.best_backend = "metal";
    hw.recommended_ngl = 99;
}
#endif

// ─── Linux detection ────────────────────────────────────────────────────────

#ifdef __linux__
static void detect_linux(HardwareProfile& hw) {
    hw.os = "linux";
    hw.arch = run_command("uname -m");
    // Trim newline
    if (!hw.arch.empty() && hw.arch.back() == '\n') hw.arch.pop_back();

    // CPU from /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                hw.cpu = line.substr(pos + 2);
            }
            break;
        }
    }

    hw.cpu_cores = static_cast<int32_t>(std::thread::hardware_concurrency());

    // CPU features
    std::ifstream cpuinfo2("/proc/cpuinfo");
    while (std::getline(cpuinfo2, line)) {
        if (line.find("flags") != std::string::npos) {
            if (line.find("avx512") != std::string::npos) hw.cpu_features = "avx512";
            else if (line.find("avx2") != std::string::npos) hw.cpu_features = "avx2";
            else hw.cpu_features = "sse4";
            break;
        }
    }

    // RAM from /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal") != std::string::npos) {
            int64_t kb = 0;
            sscanf(line.c_str(), "MemTotal: %ld kB", &kb);
            hw.ram_gb = kb / (1024 * 1024);
            break;
        }
    }

    // NVIDIA GPU
    std::string nvidia_smi = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null");
    if (!nvidia_smi.empty()) {
        std::istringstream ss(nvidia_smi);
        std::string gpu_name, vram_str;
        while (std::getline(ss, gpu_name, ',')) {
            std::getline(ss, vram_str, '\n');

            GPUInfo gpu;
            gpu.vendor = "NVIDIA";
            gpu.is_discrete = true;

            // Trim
            size_t s = gpu_name.find_first_not_of(" \t");
            size_t e = gpu_name.find_last_not_of(" \t");
            if (s != std::string::npos) gpu.name = gpu_name.substr(s, e - s + 1);

            // Parse VRAM (e.g., "24576 MiB")
            long long mb = 0;
            if (sscanf(vram_str.c_str(), "%lld MiB", &mb) == 1) {
                gpu.vram_mb = mb;
            }

            hw.gpus.push_back(gpu);
        }
        hw.best_backend = "cuda";
        hw.recommended_ngl = 99;
    }

    // AMD GPU (ROCm)
    if (hw.gpus.empty()) {
        std::string rocminfo = run_command("rocminfo 2>/dev/null | grep 'Marketing Name'");
        if (!rocminfo.empty()) {
            GPUInfo gpu;
            gpu.vendor = "AMD";
            gpu.is_discrete = true;
            size_t pos = rocminfo.find(':');
            if (pos != std::string::npos) {
                gpu.name = rocminfo.substr(pos + 2);
                if (!gpu.name.empty() && gpu.name.back() == '\n') gpu.name.pop_back();
            }
            hw.gpus.push_back(gpu);
            hw.best_backend = "hip";
            hw.recommended_ngl = 99;
        }
    }

    // No GPU
    if (hw.gpus.empty()) {
        hw.best_backend = "cpu";
        hw.recommended_ngl = 0;
    }
}
#endif

// ─── Windows detection ──────────────────────────────────────────────────────

#ifdef _WIN32
#include <windows.h>

static void detect_windows(HardwareProfile& hw) {
    hw.os = "windows";

    // Architecture
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    hw.arch = (si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64) ? "x86_64" : "x86";

    // CPU
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0,
        KEY_READ, &hKey) == ERROR_SUCCESS) {
        char buf[256];
        DWORD size = sizeof(buf);
        if (RegQueryValueExA(hKey, "ProcessorNameString", nullptr, nullptr,
                            (LPBYTE)buf, &size) == ERROR_SUCCESS) {
            hw.cpu = buf;
        }
        RegCloseKey(hKey);
    }

    hw.cpu_cores = static_cast<int32_t>(std::thread::hardware_concurrency());

    // RAM
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        hw.ram_gb = mem.ullTotalPhys / (1024ULL * 1024ULL * 1024ULL);
    }

    // GPU detection via WMI or NVML
    // For now, assume NVIDIA if available
    std::string nvml_check = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul");
    if (!nvml_check.empty()) {
        GPUInfo gpu;
        gpu.vendor = "NVIDIA";
        gpu.is_discrete = true;
        size_t pos = nvml_check.find(',');
        if (pos != std::string::npos) {
            gpu.name = nvml_check.substr(0, pos);
        }
        hw.gpus.push_back(gpu);
        hw.best_backend = "cuda";
        hw.recommended_ngl = 99;
    } else {
        hw.best_backend = "cpu";
        hw.recommended_ngl = 0;
    }
}
#endif

// ─── Public API ─────────────────────────────────────────────────────────────

HardwareProfile detect_hardware() {
    HardwareProfile hw;

#ifdef __APPLE__
    detect_macos(hw);
#elif defined(__linux__)
    detect_linux(hw);
#elif defined(_WIN32)
    detect_windows(hw);
#endif

    return hw;
}

int32_t auto_ngl(const HardwareProfile& hw) {
    return hw.recommended_ngl;
}

std::string auto_backend(const HardwareProfile& hw) {
    return hw.best_backend;
}

void HardwareProfile::print() const {
    std::cout << "[anvil] Detected hardware:\n";
    std::cout << "  OS:        " << os << " " << arch << "\n";
    std::cout << "  CPU:       " << cpu << " (" << cpu_cores << " cores)\n";
    std::cout << "  RAM:       " << ram_gb << " GB\n";

    for (const auto& gpu : gpus) {
        std::cout << "  GPU:       " << gpu.name;
        if (gpu.vram_mb > 0) {
            std::cout << " (" << (gpu.vram_mb / 1024) << " GB)";
        }
        std::cout << "\n";
    }

    std::cout << "  Backend:   " << best_backend << " (auto-selected)\n";
    std::cout << "  Offload:   " << recommended_ngl << " layers\n";
}

} // namespace anvil
