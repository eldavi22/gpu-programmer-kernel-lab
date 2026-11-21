#include "gpu_gemm.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
int main() {
  gpu_gemm::Matrix a(2,3), b(3,2); a.values={1,2,3,4,5,6}; b.values={7,8,9,10,11,12};
  const auto c=gpu_gemm::multiply_cpu(a,b); assert(c.rows==2 && c.cols==2);
  assert(std::fabs(c.at(0,0)-58.0f)<1e-6f); assert(std::fabs(c.at(0,1)-64.0f)<1e-6f);
  assert(std::fabs(c.at(1,0)-139.0f)<1e-6f); assert(std::fabs(c.at(1,1)-154.0f)<1e-6f);
  const auto check=gpu_gemm::validate(c,c); assert(check.passed && check.mismatches==0); std::cout << "test_cpu: PASS\n";
}

