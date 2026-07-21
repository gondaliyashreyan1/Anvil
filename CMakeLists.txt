cmake_minimum_required(VERSION 3.22)
project(anvil C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Build the llama.cpp backend as a static library ---
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/backends/llama-turbo)

# Override llama.cpp options: we only need the static lib, no examples/tests/server
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Platform-specific backend flags
if(APPLE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
elseif(UNIX)
    # Linux: try Vulkan, fallback to CPU
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(${LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama-build EXCLUDE_FROM_ALL)

# --- FTXUI (terminal UI library) ---
include(FetchContent)
FetchContent_Declare(ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
    GIT_TAG        v7.0.1
    GIT_SHALLOW    TRUE
)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ftxui)

# --- Build anvil ---
add_executable(anvil anvil/src/anvil.cpp)

target_include_directories(anvil PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)

target_link_libraries(anvil PRIVATE llama ftxui::ftxui)

# Platform-specific link flags
if(APPLE)
    target_link_libraries(anvil PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework Accelerate"
        "-framework QuartzCore"
        "-framework CoreFoundation"
        "-framework IOKit"
        "-lc++"
        "-lobjc"
    )
elseif(UNIX)
    target_link_libraries(anvil PRIVATE pthread dl m)
    # Vulkan if available
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        target_link_libraries(anvil PRIVATE Vulkan::Vulkan)
    endif()
endif()

# --- Install ---
install(TARGETS anvil RUNTIME DESTINATION bin)
