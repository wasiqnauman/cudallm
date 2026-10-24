cmake_minimum_required(VERSION 3.24)
project(cudallm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# No --use_fast_math anywhere (DESIGN.md §8).
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra")

find_package(CUDAToolkit REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

file(GLOB CUDALLM_CORE_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/common/*.cpp ${CMAKE_SOURCE_DIR}/src/common/*.cu
    ${CMAKE_SOURCE_DIR}/src/model/*.cpp ${CMAKE_SOURCE_DIR}/src/model/*.cu
    ${CMAKE_SOURCE_DIR}/src/engine/*.cpp ${CMAKE_SOURCE_DIR}/src/engine/*.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/*.cpp ${CMAKE_SOURCE_DIR}/src/kernels/*.cu)

add_library(cudallm_core STATIC ${CUDALLM_CORE_SOURCES})
set_target_properties(cudallm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(cudallm_core PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(cudallm_core PUBLIC CUDA::cudart)

add_library(cudallm SHARED ${CMAKE_SOURCE_DIR}/src/capi/cudallm_capi.cpp)
target_link_libraries(cudallm PRIVATE cudallm_core)

foreach(tool cudallm_bench cudallm_kernelbench cudallm_bandwidth)
    add_executable(${tool} ${CMAKE_SOURCE_DIR}/src/tools/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE cudallm_core)
endforeach()
