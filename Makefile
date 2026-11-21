CXX ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
INCLUDES := -Iinclude
BUILD := build
.PHONY: all cpu cuda test clean
all: cpu
cpu: $(BUILD)/gpu-gemm
$(BUILD):
	mkdir -p $(BUILD)
$(BUILD)/cpu_gemm.o: src/cpu_gemm.cpp include/gpu_gemm.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD)/main.o: src/main.cpp include/gpu_gemm.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD)/gpu-gemm: $(BUILD)/cpu_gemm.o $(BUILD)/main.o
	$(CXX) $(CXXFLAGS) $^ -o $@
cuda: $(BUILD)
	$(NVCC) -O3 -std=c++17 -DGPU_GEMM_ENABLE_CUDA $(INCLUDES) -c src/cpu_gemm.cpp -o $(BUILD)/cpu_gemm.o
	$(NVCC) -O3 -std=c++17 -DGPU_GEMM_ENABLE_CUDA $(INCLUDES) -c src/main.cpp -o $(BUILD)/main.o
	$(NVCC) -O3 -std=c++17 -DGPU_GEMM_ENABLE_CUDA $(INCLUDES) -c src/cuda_gemm.cu -o $(BUILD)/cuda_gemm.o
	$(NVCC) -O3 $(BUILD)/cpu_gemm.o $(BUILD)/main.o $(BUILD)/cuda_gemm.o -o $(BUILD)/gpu-gemm
test: $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDES) src/cpu_gemm.cpp tests/test_cpu.cpp -o $(BUILD)/test_cpu
	$(BUILD)/test_cpu
clean:
	rm -rf $(BUILD)

