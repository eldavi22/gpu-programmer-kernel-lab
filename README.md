# GPU Programmer Kernel Lab

A focused CUDA/C++ performance lab for tiled matrix multiplication. The project demonstrates the complete shape of a GPU programming task: deterministic inputs, a CPU reference implementation, a shared-memory CUDA kernel, host-side orchestration, CUDA event timing, numerical validation, build automation, and profiling instructions.

## What it demonstrates

- A tiled GEMM kernel written in CUDA C++ using shared memory and coalesced global-memory accesses.
- Host-side C++17 integration with explicit allocation, asynchronous copies, streams, events, launch configuration, and cleanup.
- Runtime-selectable tile sizes of 8, 16, or 32 threads per dimension.
- A deterministic CPU reference path and absolute/relative error validation.
- CPU-only builds for machines without the CUDA toolkit, plus a CUDA build path for NVIDIA systems.
- Reproducible unit tests and a GitHub Actions CPU build.

## Project layout

```text
include/gpu_gemm.hpp       Public data structures and benchmark API
src/cpu_gemm.cpp           Deterministic data, CPU reference, validation, timing
src/cuda_gemm.cu           Shared-memory tiled CUDA kernel and host integration
src/main.cpp               CLI, benchmark reporting, backend selection
tests/test_cpu.cpp         Reference GEMM and validation tests
docs/profiling.md          Nsight Systems and Nsight Compute commands
Makefile                  Minimal compiler-driven build
CMakeLists.txt             CMake build for CPU or CUDA configurations
```

## Requirements

CPU path: C++17 compiler and GNU Make.

CUDA path: NVIDIA GPU with a supported compute capability and the CUDA Toolkit with `nvcc`.

## Build and run locally

The CPU path works on any C++17 machine:

```bash
make cpu
make test
./build/gpu-gemm --backend cpu --m 256 --n 256 --k 256 --iterations 3
```

On a CUDA-enabled host:

```bash
make cuda
./build/gpu-gemm --backend cuda --m 1024 --n 1024 --k 1024 --iterations 20 --tile 16
```

The `auto` backend uses CUDA when the binary was compiled with CUDA support; otherwise it reports the CPU build explicitly. Requesting `--backend cuda` from a CPU-only binary fails clearly instead of silently claiming GPU execution.

### CMake

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure

cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
cmake --build build-cuda -j
```

## Benchmark output

The executable prints average milliseconds per iteration, calculated CPU GFLOP/s, calculated CUDA TFLOP/s, tile size, and validation status. The FLOP calculation is `2 * M * N * K`. CUDA timing uses CUDA events recorded on the same stream as the kernel launches.

This repository does not hard-code a performance claim: GPU throughput depends on the NVIDIA model, clocks, driver, CUDA version, matrix shape, tile size, and thermal state. Use the commands in [`docs/profiling.md`](docs/profiling.md) to collect device-specific evidence.

## Correctness contract

Every CUDA result is copied back to the host and compared with the CPU reference using both absolute and relative tolerances. The executable exits non-zero on a validation failure. Boundary checks in the kernel support matrix dimensions that are not multiples of the selected tile size.

## Profiling

See [`docs/profiling.md`](docs/profiling.md) for Nsight Systems and Nsight Compute commands, plus the metadata that should accompany a performance result.

## License

MIT. See [`LICENSE`](LICENSE).
