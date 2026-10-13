# syntax=docker/dockerfile:1

# CUDA 13.3 or newer. Do not go below it: nvcc in 13.1 does not know -std=c++23, which
# CMakeLists.txt passes to nvcc by hand because CMake has no CUDA 23 dialect, and the build then
# fails with "nvcc fatal : Value 'c++23' is not defined for option 'std'" on the first .cu file.
FROM nvidia/cuda:13.3.1-devel-ubuntu24.04 AS build

# Two toolchain constraints, both from the tree's C++23 use (docs/engineering.md):
#  * <flat_map> is libstdc++ 15 -- Ubuntu 24.04 ships 13, and 14 at newest -- so g++-15 comes from the
#    ubuntu-toolchain-r PPA. It becomes the *default* c++/cc, not just CMake's compiler variables,
#    because CMake's CUDA dyndep scan runs /usr/bin/c++ directly and ignores CMAKE_CUDA_HOST_COMPILER.
#  * CMake 3.28 and newer scan C++23 sources for modules by default, and gcc's -fmodules-ts scan
#    cannot see <flat_map> either, so the scan is switched off at configure time.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        cmake \
        gnupg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
        software-properties-common \
    && add-apt-repository --yes ppa:ubuntu-toolchain-r/test \
    && apt-get install --yes --no-install-recommends g++-15 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-15 200 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-15 200 \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-15 CXX=g++-15

WORKDIR /src
COPY . .

RUN cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++-15 \
        -DCMAKE_CUDA_HOST_COMPILER=g++-15 \
        -DCMAKE_CUDA_FLAGS="-ccbin g++-15" \
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
