# syntax=docker/dockerfile:1

# CUDA 13.3 or newer. Do not go below it: nvcc in 13.1 does not know -std=c++23, which
# CMakeLists.txt passes to nvcc by hand because CMake has no CUDA 23 dialect, and the build then
# fails with "nvcc fatal : Value 'c++23' is not defined for option 'std'" on the first .cu file.
FROM nvidia/cuda:13.3.1-devel-ubuntu24.04 AS build

# Ubuntu 24.04 defaults to gcc 13, whose libstdc++ has neither <flat_map> nor <print>; the tree uses
# both (docs/engineering.md). The default c++/cc are pointed at 14 rather than relying on CMake's
# compiler variables, because CMake's CUDA dyndep scan runs /usr/bin/c++ directly and ignores
# CMAKE_CUDA_HOST_COMPILER -- which is where flat_map was still coming from gcc 13.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        cmake \
        g++-14 \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-14 200 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-14 200 \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14

WORKDIR /src
COPY . .

# CMAKE_CXX_SCAN_FOR_MODULES=OFF: CMake 3.28 and newer scan C++23 sources for modules by default, and
# gcc-14's -fmodules-ts scan cannot see libstdc++'s <flat_map> either. The tree uses C++23 library
# features, not modules.
RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++-14 \
        -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
        -DCMAKE_CUDA_FLAGS="-ccbin g++-14" \
        -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel --target ninfer ninfer-serve

FROM nvidia/cuda:13.3.1-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libswscale7 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/apps/ninfer /usr/local/bin/ninfer
COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

CMD ["ninfer-serve", "--help"]
