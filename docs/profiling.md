# Profiling guide

The CUDA path uses CUDA events around the repeated kernel launches, so `cuda_ms` reports the average kernel time per iteration and excludes one-time host-side setup. Use Nsight Systems for a timeline and Nsight Compute for kernel-level counters.

```bash
make cuda
./build/gpu-gemm --backend cuda --m 1024 --n 1024 --k 1024 --iterations 20 --tile 16
nsys profile --stats=true --force-overwrite=true -o reports/gemm ./build/gpu-gemm --backend cuda --m 2048 --n 2048 --k 2048 --iterations 50
ncu --set full --target-processes all ./build/gpu-gemm --backend cuda --m 2048 --n 2048 --k 2048 --iterations 10 --tile 16
```

Compare tile sizes 8, 16, and 32 on the same device and shape. Record GPU model, driver, CUDA toolkit, matrix dimensions, tile size, iteration count, and whether transfers were included before comparing results.

