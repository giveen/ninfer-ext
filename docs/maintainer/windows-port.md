# Windows port

NInfer was Linux-only: no `_WIN32` appeared anywhere in `src/`, `apps/` or `include/`. This reference
records what the Windows build is, which constraints are hard floors, how the platform seam is drawn,
and — the larger half — what has **not** been established yet.

Behavior on Windows hardware is not yet verified by the maintainers. The
[compile gate](#the-compile-gate) proves the tree builds with MSVC; everything about how it *runs* on
a Windows machine is outstanding, and the sections [Not established](#not-established) and
[Reporting a result](#reporting-a-result) exist so that a tester's run can close them.

## What it builds

`ninfer-serve` (HTTP server) and `ninfer` (CLI), as x64 executables, from the same sources and the
same CMake project as Linux, with the same feature set — text, vision and remote media input. The
CI gate builds those two; `ninfer-perplexity`, `ninfer-quantize` and `ninfer-sensitivity` are targets
in the same project and share the same platform layer, but the gate does not build them yet.

## Toolchain floors

| Requirement | Why it is a floor |
|---|---|
| **MSVC 1951+ (Visual Studio 2026 18.x)** | `<flat_map>`, which `src/artifact/schema.h` includes, shipped in MSVC 14.51. The Visual Studio 2022 toolset is 193x and cannot compile this tree at all. `CMakeLists.txt` rejects it at configure time with that reason. |
| **CUDA 13.3** | `sm_120a` is the only supported target. The CUDA 13.3 Windows installation guide lists MSVC 195x / Visual Studio 2026 18.x as supported, which is the same window `<flat_map>` needs. |
| **CMake 3.28+ and Ninja** | Unchanged from Linux; Ninja keeps the single-slot link pool that the device link needs. |
| **vcpkg (`vcpkg.json`)** | FFmpeg and libcurl have no system packages on Windows. The manifest pins a `builtin-baseline`, so the versions are fixed by the repository rather than by the runner's vcpkg checkout. |

Device code is compiled below the host standard. nvcc's own `--std` accepts at most `c++20` (CUDA
13.3 documents `--std=c++03..20`), so `.cu` translation units are C++20 while host units are
`/std:c++latest`. That split is safe only while no `.cu` includes a C++23 library header. Keep it that
way: the failure mode is a missing header inside a device unit, which reads like a toolchain problem.

## The platform seam

Two primitives appear in several places each and fail *silently* when they are wrong, so they live in
one file each and no call site names an OS API:

| Seam | Contents | Callers |
|---|---|---|
| `src/core/native_file.{h,cpp}` | `NativeFileHandle`, `open_read_only(path, unbuffered)`, `close_file`, `file_bytes`, `is_regular_file`, `read_at`, `advise_random_access` | the artifact reader, the n-gram row reader |
| `src/core/host_memory.h` | `aligned_host_alloc` / `aligned_host_free`, `advise_large_pages`, `prefault_host_pages` | the pinned host buffers, the n-gram staging buffer |

`read_at` is the interesting one. POSIX gets `pread(2)`; Windows needs a `FILE_FLAG_OVERLAPPED` handle
so that the offset travels in the `OVERLAPPED` structure instead of through a shared file pointer. That
is not a style choice: the artifact reader loads weights on several threads at once and the n-gram
reader keeps sixteen reads outstanding, so concurrent positioned reads are a contract. `read_at`
retries `EINTR` internally, returns 0 only at end of file, and reports everything else as
`std::system_error` with the platform's error category, so `what()` reads `<operation>: <message>` on
both platforms and call sites format identical text.

Everything smaller than that stays in place behind `#if defined(_WIN32)`, with the Linux path
untouched: the console/logging calls (`isatty`, `TIOCGWINSZ`, `localtime_r`), `getpid`, the
`/proc/meminfo` host-memory probe, the socket keep-alive options, the console control handler that
stands in for `SIGTERM`, Winsock initialization for media acquisition, and the tests' temporary
directories and CRT calls.

## Platform differences that change behavior, not just spelling

- **`TCP_USER_TIMEOUT` does not exist on Windows.** On Linux it, together with the 5 s SSE heartbeat,
  bounds how long a dead streaming peer goes undetected. On Windows the send timeout that the HTTP
  layer already sets and the same heartbeat carry that, so a dead peer is noticed on the next write
  rather than by a timer.
- **`/proc/meminfo` has no equivalent.** The automatic n-gram residency heuristic reads available host
  memory; on Windows it uses `GlobalMemoryStatusEx`. If the quantity cannot be read, the heuristic
  keeps its existing safe fallback: stream the table instead of mapping it.
- **`O_DIRECT` becomes `FILE_FLAG_NO_BUFFERING`,** which enforces the volume's sector alignment on
  offsets *and* lengths and reports end of file differently. The n-gram reader's contract — a short read
  is legal past the volume's declared extent and a hole inside it is corruption — is unchanged.
- **Large pages are not requested.** The pinned path asks for huge pages on Linux; on Windows that
  needs a privilege and a different allocator, so the advice is dropped. The buffer is correct either
  way; only the number of translation entries behind the pin differs.
- **`long double` is 64-bit on MSVC, not 80-bit extended.** The op oracles accumulate reference values
  in `long double`; on Windows the oracle is therefore less precise than on Linux. Contraction is the
  other half of that contract and it *does* hold: `/fp:precise` has not generated contractions by
  default since Visual Studio 2022 17.0, and `tests/cmake/NinferTests.cmake` states the option
  explicitly. Whether the calibrated tolerances still hold at the lower precision is part of what a
  tester's run has to show.

## The compile gate

`.github/workflows/windows.yml` runs on `windows-2025-vs2026` (Visual Studio 2026 18.x, MSVC 195x),
installs CUDA 13.3, solves FFmpeg and libcurl through vcpkg, configures with Ninja, builds `ninfer` and
`ninfer-serve`, and uploads a **runnable package**: the executables plus the CUDA runtime and the vcpkg
DLLs they load. It runs no tests. GitHub-hosted runners have no GPU, and the hosted GPU runners are T4
(`sm_75`), which this engine does not support — so the gate can only prove compilation, and it says so
in its header.

Anyone editing the Windows code paths locally should run
`tools/verify/win32_syntax_check.py <paths>`: it compiles the same translation units with mingw-w64,
which catches POSIX leftovers, wrong Win32 signatures and branches that never compile in seconds rather
than in a CI round trip. It is a smoke check, not MSVC.

## Not established

These are the questions a Windows machine has to answer, in the order they matter:

1. **Can a kernel dereference pinned host memory at the same address under WDDM?** The engine gates
   the registered-memory path on `cudaDevAttrCanUseHostPointerForRegisteredMem`
   (`src/core/arena.cu`), and two shipped features depend on the answer: the Gemma 4 token-embedding
   gather reads a pinned host table from a kernel through unified addressing, and the Qwen4 expert
   pager copies with direction-inferred `cudaMemcpyDefault`. If this is unsupported, those two become a
   *representation* decision — the table has to be device-resident — and not a one-line fix. The
   `cudaMallocHost` fallback in the code does not settle it, because under unified addressing it is the
   same mechanism.
2. **What does WDDM cost?** A GeForce card cannot run in TCC mode, so this is WDDM-only: per-launch
   overhead against a decode loop that issues hundreds of launches per token. The published numbers are
   Linux numbers and do not transfer.
3. **Does a long prefill kernel survive TDR?** Windows resets a GPU whose kernel runs past a few
   seconds by default. Prefill is exactly that shape. The mitigation is a registry change on the
   tester's machine (`TdrDelay`) or splitting kernels — a deployment decision, not a bug.
4. **Does the test suite pass?** No Windows machine has run `ctest --preset fast` or the real-model
   suite yet. The test sources have been made portable and the oracle's floating-point contract has an
   MSVC form, but neither has executed.

## Reporting a result

`ninfer-probe` (no model artifact needed) reports the facts these questions need: the device and
driver, `cudaDevAttrCanUseHostPointerForRegisteredMem`, and a functional test that registers host
memory, reads it from a kernel, and validates a direction-inferred copy. Its exit code is 0 when that
path works end to end. A report containing its output, the Windows and driver versions, and either a
`ctest --preset fast` summary or a `ninfer-serve` request/response transcript is what this port needs to
move from "builds" to "verified".

Running the suite on Windows needs two things the command line on Linux gets for free:

- **`jsonschema`** in the interpreter CMake found. `ninfer_json_schema_oracle_test` is a registered
  CTest entry that imports it (`tests/text/test_json_schema.py`), so it fails rather than skips without
  it: `py -3 -m pip install jsonschema`.
- **The interpreter's name.** Windows installs no `python3` entry point, so configure with an explicit
  one — `cmake -S . -B build -G Ninja -DPython3_EXECUTABLE=...` — and use `py -3` in the commands that
  [tests/README.md](../../tests/README.md) spells as `python3`.

The real-model subset is unchanged: it reads its artifact and inputs from the environment
([real-model tests](real-model-tests.md)), and the tests that need inputs the repository does not carry
skip themselves when those variables are unset.
