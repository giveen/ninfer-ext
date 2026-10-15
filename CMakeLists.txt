cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# <flat_map>, which the artifact schema uses, shipped in MSVC 14.51 (Visual Studio 2026 18.x); the
# Visual Studio 2022 toolset is 193x and cannot compile this tree at all. Fail here, with the reason,
# instead of deep inside the artifact schema.
if(MSVC AND MSVC_VERSION LESS 1951)
  message(FATAL_ERROR
    "NInfer needs MSVC 1951 or newer (Visual Studio 2026 18.x): <flat_map> is C++23 and the artifact "
    "schema depends on it. Found MSVC_VERSION=${MSVC_VERSION}.")
endif()

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Device code is compiled below the host standard, because nvcc's own --std accepts at most c++20
# (CUDA 13.3 documents --std=c++03..20). That split is safe only because no .cu translation unit
# includes a C++23 library header; keep it that way. CMake (through 4.3) has no CUDA23 dialect, so
# request CUDA20 to make CMake accept the targets, blank its standard flag and pass the newest one
# nvcc does accept directly: c++23 for a GCC/Clang host, c++20 under MSVC, which nvcc forwards as
# /std:c++20 to cl.exe for the host pass of a .cu. Without the explicit flag those units would fall
# to nvcc's default, which is older than the C++20 the device and host code is written against.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-std=c++23>)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
