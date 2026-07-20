cmake_minimum_required(VERSION 3.22)
project(anvil VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Build options ───────────────────────────────────────────────────────────

option(ANVIL_BUILD_TESTS "Build tests" OFF)
option(ANVIL_BUILD_SERVER "Build API server" OFF)

# ─── Paths ───────────────────────────────────────────────────────────────────

set(LLAMA_TURBO_DIR "${CMAKE_SOURCE_DIR}/backends/llama-turbo")
set(LLAMA_TURBO_BUILD "${CMAKE_BINARY_DIR}/llama-turbo-build")

# ─── Build llama-turbo as a subproject ───────────────────────────────────────

# We build llama-turbo as a static library with TurboQuant, MTP, NextN enabled
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)   # Suppress OpenMP warning on macOS

# Enable backends (use GGML_ prefix to avoid deprecation warnings)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(LLAMA_METAL ON CACHE BOOL "" FORCE)     # Also set legacy flag for compatibility

# Build the llama-turbo library
add_subdirectory("${LLAMA_TURBO_DIR}" "${LLAMA_TURBO_BUILD}")

# ─── Anvil sources ───────────────────────────────────────────────────────────

set(ANVIL_SOURCES
    src/main.cpp
    src/engine.cpp
    src/hardware.cpp
    src/config.cpp
)

add_executable(anvil ${ANVIL_SOURCES})

# ─── Include directories ─────────────────────────────────────────────────────

target_include_directories(anvil PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${LLAMA_TURBO_DIR}/include
    ${LLAMA_TURBO_DIR}/common
    ${LLAMA_TURBO_DIR}/ggml/include
)

# ─── Link libraries ──────────────────────────────────────────────────────────

target_link_libraries(anvil PRIVATE
    llama
    ggml
    ggml-base
)

# Platform-specific linking
if(APPLE)
    target_link_libraries(anvil PRIVATE
        "-framework Metal"
        "-framework Accelerate"
        "-framework Foundation"
        "-framework QuartzCore"
        "-framework CoreFoundation"
    )
    target_compile_definitions(anvil PRIVATE ANVIL_PLATFORM_MACOS)
elseif(UNIX)
    target_compile_definitions(anvil PRIVATE ANVIL_PLATFORM_LINUX)
    target_link_libraries(anvil PRIVATE pthread dl m)
    target_link_libraries(anvil PRIVATE stdc++fs)
elseif(WIN32)
    target_compile_definitions(anvil PRIVATE ANVIL_PLATFORM_WINDOWS)
    target_link_libraries(anvil PRIVATE ws2_32)
endif()

# ─── Compiler flags ──────────────────────────────────────────────────────────

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(anvil PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -O2
    )
    # Enable optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(anvil PRIVATE -O3 -march=native)
    endif()
endif()

if(MSVC)
    target_compile_options(anvil PRIVATE /W4 /O2)
endif()

# ─── Install ─────────────────────────────────────────────────────────────────

install(TARGETS anvil DESTINATION bin)

# ─── Print configuration ─────────────────────────────────────────────────────

message(STATUS "Anvil v${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  llama-turbo: ${LLAMA_TURBO_DIR}")
