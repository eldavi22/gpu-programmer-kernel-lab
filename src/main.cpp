#include "gpu_gemm.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
struct Options { std::size_t m=256, n=256, k=256; int iterations=3, tile=16; std::string backend="auto"; };
void usage(const char* binary) { std::cout << "Usage: " << binary << " [--m M] [--n N] [--k K] [--iterations I] [--tile 8|16|32] [--backend auto|cpu|cuda]\n"; }
std::size_t parse_size(const std::string& value, const char* name) {
  const long long parsed = std::stoll(value);
  if (parsed <= 0) throw std::invalid_argument(std::string(name) + " must be positive");
  return static_cast<std::size_t>(parsed);
}
}

int main(int argc, char** argv) {
  Options options;
  try {
    for (int i=1; i<argc; ++i) {
      const std::string arg=argv[i];
      if (arg=="--help") { usage(argv[0]); return 0; }
      if (i+1>=argc) throw std::invalid_argument("missing value for " + arg);
      const std::string value=argv[++i];
      if (arg=="--m") options.m=parse_size(value,"M");
      else if (arg=="--n") options.n=parse_size(value,"N");
      else if (arg=="--k") options.k=parse_size(value,"K");
      else if (arg=="--iterations") options.iterations=static_cast<int>(parse_size(value,"iterations"));
      else if (arg=="--tile") options.tile=static_cast<int>(parse_size(value,"tile"));
      else if (arg=="--backend") options.backend=value;
      else throw std::invalid_argument("unknown option " + arg);
    }
    if (options.backend!="auto" && options.backend!="cpu" && options.backend!="cuda") throw std::invalid_argument("backend must be auto, cpu, or cuda");
    if (options.tile!=8 && options.tile!=16 && options.tile!=32) throw std::invalid_argument("tile must be 8, 16, or 32");
    const auto a=gpu_gemm::make_deterministic_matrix(options.m,options.k,17);
    const auto b=gpu_gemm::make_deterministic_matrix(options.k,options.n,31);
    const auto reference=gpu_gemm::multiply_cpu(a,b);
    gpu_gemm::Matrix cpu_output(options.m,options.n);
    const double cpu_ms=gpu_gemm::benchmark_cpu(a,b,cpu_output,options.iterations);
    const auto cpu_check=gpu_gemm::validate(reference,cpu_output);
    const double operations=2.0*static_cast<double>(options.m)*options.n*options.k;
    std::cout << std::fixed << std::setprecision(4)
              << "gpu-programmer-kernel-lab\nshape=" << options.m << "x" << options.k << " * " << options.k << "x" << options.n
              << " iterations=" << options.iterations << "\ncpu_ms=" << cpu_ms << " cpu_GFLOP/s=" << operations/(cpu_ms*1e6)
              << " validation=" << (cpu_check.passed?"PASS":"FAIL") << "\n";
#if defined(GPU_GEMM_ENABLE_CUDA)
    if (options.backend!="cpu") {
      const auto cuda=gpu_gemm::benchmark_cuda(a,b,reference,options.iterations,options.tile);
      std::cout << "cuda_ms=" << cuda.milliseconds << " cuda_TFLOP/s=" << cuda.tera_flops << " tile=" << options.tile
                << " validation=" << (cuda.validation.passed?"PASS":"FAIL") << "\n";
      return cuda.validation.passed?0:2;
    }
#else
    if (options.backend=="cuda") { std::cerr << "CUDA backend was not compiled. Run 'make cuda' with the CUDA toolkit installed.\n"; return 3; }
    if (options.backend=="auto") std::cout << "cuda=unavailable (CPU build)\n";
#endif
    return cpu_check.passed?0:2;
  } catch (const std::exception& error) { std::cerr << "error: " << error.what() << "\n"; usage(argv[0]); return 1; }
}

