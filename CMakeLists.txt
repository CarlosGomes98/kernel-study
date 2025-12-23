cmake_minimum_required(VERSION 3.19)
project(KERNEL_STUDY LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)

# Detect GPU architecture
if(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
    execute_process(
        COMMAND bash -c "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.'"
        OUTPUT_VARIABLE CUDA_COMPUTE_CAPABILITY
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(CUDA_COMPUTE_CAPABILITY STREQUAL "")
        set(CUDA_COMPUTE_CAPABILITY 86)
        message(WARNING "Could not detect GPU architecture, defaulting to sm_86")
    else()
        message(STATUS "Detected GPU architecture: sm_${CUDA_COMPUTE_CAPABILITY}")
    endif()
endif()

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# add_compile_options("--ptxas-options=-v")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(benchmark benchmark.cu ${SRC})
set_target_properties(benchmark PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(benchmark ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

