#include "gpu_gemm.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace gpu_gemm { namespace {
void check_cuda(cudaError_t status, const char* operation) { if (status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status)); }

template<int TILE> __global__ void tiled_gemm(const float* a,const float* b,float* c,int m,int n,int k) {
  __shared__ float tile_a[TILE][TILE]; __shared__ float tile_b[TILE][TILE];
  const int row=blockIdx.y*TILE+threadIdx.y, col=blockIdx.x*TILE+threadIdx.x; float accumulator=0.0f;
  for (int tile=0; tile<(k+TILE-1)/TILE; ++tile) {
    const int a_col=tile*TILE+threadIdx.x, b_row=tile*TILE+threadIdx.y;
    tile_a[threadIdx.y][threadIdx.x]=(row<m && a_col<k)?a[row*k+a_col]:0.0f;
    tile_b[threadIdx.y][threadIdx.x]=(b_row<k && col<n)?b[b_row*n+col]:0.0f;
    __syncthreads();
    #pragma unroll
    for (int j=0;j<TILE;++j) accumulator+=tile_a[threadIdx.y][j]*tile_b[j][threadIdx.x];
    __syncthreads();
  }
  if (row<m && col<n) c[row*n+col]=accumulator;
}
template<int TILE> void launch(const float* a,const float* b,float* c,int m,int n,int k,cudaStream_t stream) {
  tiled_gemm<TILE><<<dim3((n+TILE-1)/TILE,(m+TILE-1)/TILE),dim3(TILE,TILE),0,stream>>>(a,b,c,m,n,k);
}
} // namespace

BenchmarkResult benchmark_cuda(const Matrix& a,const Matrix& b,const Matrix& reference,int iterations,int tile_size) {
  if (a.cols!=b.rows) throw std::invalid_argument("matrix dimensions are not compatible");
  const std::size_t a_bytes=a.values.size()*sizeof(float), b_bytes=b.values.size()*sizeof(float), c_bytes=a.rows*b.cols*sizeof(float);
  float *device_a=nullptr,*device_b=nullptr,*device_c=nullptr; cudaStream_t stream=nullptr; cudaEvent_t start=nullptr,stop=nullptr;
  check_cuda(cudaMalloc(&device_a,a_bytes),"cudaMalloc(A)"); check_cuda(cudaMalloc(&device_b,b_bytes),"cudaMalloc(B)"); check_cuda(cudaMalloc(&device_c,c_bytes),"cudaMalloc(C)");
  check_cuda(cudaStreamCreate(&stream),"cudaStreamCreate"); check_cuda(cudaEventCreate(&start),"cudaEventCreate(start)"); check_cuda(cudaEventCreate(&stop),"cudaEventCreate(stop)");
  try {
    check_cuda(cudaMemcpyAsync(device_a,a.values.data(),a_bytes,cudaMemcpyHostToDevice,stream),"copy A"); check_cuda(cudaMemcpyAsync(device_b,b.values.data(),b_bytes,cudaMemcpyHostToDevice,stream),"copy B"); check_cuda(cudaStreamSynchronize(stream),"upload synchronize");
    check_cuda(cudaEventRecord(start,stream),"event start");
    for(int i=0;i<iterations;++i) { if(tile_size==8) launch<8>(device_a,device_b,device_c,a.rows,b.cols,a.cols,stream); else if(tile_size==32) launch<32>(device_a,device_b,device_c,a.rows,b.cols,a.cols,stream); else launch<16>(device_a,device_b,device_c,a.rows,b.cols,a.cols,stream); }
    check_cuda(cudaEventRecord(stop,stream),"event stop"); check_cuda(cudaEventSynchronize(stop),"event synchronize"); float elapsed_ms=0.0f; check_cuda(cudaEventElapsedTime(&elapsed_ms,start,stop),"event elapsed time");
    Matrix actual(a.rows,b.cols); check_cuda(cudaMemcpy(actual.values.data(),device_c,c_bytes,cudaMemcpyDeviceToHost),"download C"); const auto validation=validate(reference,actual); const double per_iteration_ms=elapsed_ms/iterations; const double operations=2.0*static_cast<double>(a.rows)*a.cols*b.cols;
    cudaEventDestroy(start); cudaEventDestroy(stop); cudaStreamDestroy(stream); cudaFree(device_a); cudaFree(device_b); cudaFree(device_c); return {"cuda",per_iteration_ms,operations/(per_iteration_ms*1e9),validation};
  } catch (...) { if(start) cudaEventDestroy(start); if(stop) cudaEventDestroy(stop); if(stream) cudaStreamDestroy(stream); cudaFree(device_a); cudaFree(device_b); cudaFree(device_c); throw; }
}
} // namespace gpu_gemm

