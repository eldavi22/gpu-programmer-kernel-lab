#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace gpu_gemm {

struct Matrix {
  std::size_t rows{};
  std::size_t cols{};
  std::vector<float> values;

  Matrix() = default;
  Matrix(std::size_t row_count, std::size_t column_count);
  float& at(std::size_t row, std::size_t col);
  const float& at(std::size_t row, std::size_t col) const;
};

struct ValidationResult {
  bool passed{};
  float max_abs_error{};
  float max_relative_error{};
  std::size_t mismatches{};
};

struct BenchmarkResult {
  std::string backend;
  double milliseconds{};
  double tera_flops{};
  ValidationResult validation;
};

Matrix make_deterministic_matrix(std::size_t rows, std::size_t cols, int seed);
Matrix multiply_cpu(const Matrix& a, const Matrix& b);
ValidationResult validate(const Matrix& expected, const Matrix& actual,
                          float absolute_tolerance = 1e-3f,
                          float relative_tolerance = 1e-3f);
double benchmark_cpu(const Matrix& a, const Matrix& b, Matrix& output,
                     int iterations);

#if defined(GPU_GEMM_ENABLE_CUDA)
BenchmarkResult benchmark_cuda(const Matrix& a, const Matrix& b,
                               const Matrix& reference, int iterations,
                               int tile_size);
#endif

}  // namespace gpu_gemm

