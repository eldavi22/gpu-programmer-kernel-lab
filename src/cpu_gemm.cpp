#include "gpu_gemm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace gpu_gemm {

Matrix::Matrix(std::size_t row_count, std::size_t column_count)
    : rows(row_count), cols(column_count), values(row_count * column_count, 0.0f) {}

float& Matrix::at(std::size_t row, std::size_t col) { return values[row * cols + col]; }
const float& Matrix::at(std::size_t row, std::size_t col) const { return values[row * cols + col]; }

Matrix make_deterministic_matrix(std::size_t rows, std::size_t cols, int seed) {
  Matrix matrix(rows, cols);
  std::uint32_t state = static_cast<std::uint32_t>(seed) + 0x9E3779B9u;
  for (float& value : matrix.values) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    value = (static_cast<float>(state % 2001u) / 1000.0f) - 1.0f;
  }
  return matrix;
}

Matrix multiply_cpu(const Matrix& a, const Matrix& b) {
  if (a.cols != b.rows) throw std::invalid_argument("matrix dimensions are not compatible");
  Matrix output(a.rows, b.cols);
  for (std::size_t row = 0; row < a.rows; ++row) {
    for (std::size_t k = 0; k < a.cols; ++k) {
      const float a_value = a.at(row, k);
      for (std::size_t col = 0; col < b.cols; ++col) output.at(row, col) += a_value * b.at(k, col);
    }
  }
  return output;
}

ValidationResult validate(const Matrix& expected, const Matrix& actual,
                          float absolute_tolerance, float relative_tolerance) {
  if (expected.rows != actual.rows || expected.cols != actual.cols)
    throw std::invalid_argument("cannot validate matrices with different shapes");
  ValidationResult result{true, 0.0f, 0.0f, 0};
  for (std::size_t i = 0; i < expected.values.size(); ++i) {
    const float error = std::fabs(expected.values[i] - actual.values[i]);
    const float scale = std::max(std::fabs(expected.values[i]), 1e-6f);
    const float relative_error = error / scale;
    result.max_abs_error = std::max(result.max_abs_error, error);
    result.max_relative_error = std::max(result.max_relative_error, relative_error);
    if (error > absolute_tolerance && relative_error > relative_tolerance) {
      result.passed = false; ++result.mismatches;
    }
  }
  return result;
}

double benchmark_cpu(const Matrix& a, const Matrix& b, Matrix& output, int iterations) {
  if (iterations <= 0) throw std::invalid_argument("iterations must be positive");
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) output = multiply_cpu(a, b);
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count() / iterations;
}

}  // namespace gpu_gemm

