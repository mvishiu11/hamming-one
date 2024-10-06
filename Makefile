# Compiler
CXX = g++
NVCC = nvcc

# Directories
INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
INPUT_DIR = input
OUTPUT_DIR = output

# Source files
CPP_FILES = $(wildcard $(SRC_DIR)/*.cpp)
CU_FILES = $(wildcard $(SRC_DIR)/*.cu)

# Object files (separate for CPP and CU files)
OBJ_CPP_FILES = $(addprefix $(OBJ_DIR)/, $(notdir $(CPP_FILES:.cpp=.o)))
OBJ_CU_FILES = $(addprefix $(OBJ_DIR)/, $(notdir $(CU_FILES:.cu:.o)))

# Targets
TEST_SOLUTION_TARGET = $(BIN_DIR)/test_solution
GENERATE_INPUT_TARGET = $(BIN_DIR)/generate_input
CPU_TARGET = $(BIN_DIR)/hamming_one_cpu
GPU_TARGET = $(BIN_DIR)/hamming_one_gpu
OPTIMISED_GPU_TARGET = $(BIN_DIR)/hamming_one_optimised_gpu
HASHTABLE_CPU_TARGET = $(BIN_DIR)/hamming_one_hashtable_cpu
HASHTABLE_GPU_TARGET = $(BIN_DIR)/hamming_one_hashtable_gpu

# Compiler flags
CXXFLAGS = -I$(INCLUDE_DIR) -O3
NVCCFLAGS = -I$(INCLUDE_DIR) -O3

# CUDA libs
CUDA_LIBS = -lcudart -L/usr/local/cuda/lib64

# Ensure directories exist
DIRS := $(OBJ_DIR) $(BIN_DIR) $(OUTPUT_DIR) $(INPUT_DIR)

.PHONY: dirs
dirs:
	mkdir -p $(DIRS)

# Build all executables, ensuring directories are created
all: dirs $(TEST_SOLUTION_TARGET) $(GENERATE_INPUT_TARGET) $(CPU_TARGET) $(HASHTABLE_CPU_TARGET) $(GPU_TARGET) $(OPTIMISED_GPU_TARGET) $(HASHTABLE_GPU_TARGET)

# Rules for building object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Test solution executable
$(TEST_SOLUTION_TARGET): $(OBJ_DIR)/test_solution.o
	$(CXX) $(CXXFLAGS) $(OBJ_DIR)/test_solution.o -o $(TEST_SOLUTION_TARGET)

# Generate input executable
$(GENERATE_INPUT_TARGET): $(OBJ_DIR)/generate_input.o
	$(NVCC) $(NVCCFLAGS) $(OBJ_DIR)/generate_input.o -o $(GENERATE_INPUT_TARGET) $(CUDA_LIBS)

# CPU executable
$(CPU_TARGET): $(OBJ_DIR)/hamming_one_cpu.o
	$(CXX) $(CXXFLAGS) $(OBJ_DIR)/hamming_one_cpu.o -o $(CPU_TARGET)

# Hashtable CPU executable
$(HASHTABLE_CPU_TARGET): $(OBJ_DIR)/hamming_one_hashtable_cpu.o
	$(CXX) $(CXXFLAGS) $(OBJ_DIR)/hamming_one_hashtable_cpu.o -o $(HASHTABLE_CPU_TARGET)

# GPU executable
$(GPU_TARGET): $(OBJ_DIR)/hamming_one_gpu.o
	$(NVCC) $(NVCCFLAGS) $(OBJ_DIR)/hamming_one_gpu.o -o $(GPU_TARGET) $(CUDA_LIBS)

# Optimised GPU executable
$(OPTIMISED_GPU_TARGET): $(OBJ_DIR)/hamming_one_optimised_gpu.o
	$(NVCC) $(NVCCFLAGS) $(OBJ_DIR)/hamming_one_optimised_gpu.o -o $(OPTIMISED_GPU_TARGET) $(CUDA_LIBS)

# Hashtable GPU executable
$(HASHTABLE_GPU_TARGET): $(OBJ_DIR)/hamming_one_hashtable_gpu.o
	$(NVCC) $(NVCCFLAGS) $(OBJ_DIR)/hamming_one_hashtable_gpu.o -o $(HASHTABLE_GPU_TARGET) $(CUDA_LIBS)

# Clean up
clean:
	rm -rf $(OBJ_DIR)/*.o $(BIN_DIR)/* $(OUTPUT_DIR)/*

# Run script
run: all
	./run.sh