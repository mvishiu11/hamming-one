#!/bin/bash

# Default flags
GEN_INPUT=true
RUN_CPU=true

# Parse arguments
for arg in "$@"
do
    case $arg in
        --no-gen)
        GEN_INPUT=false
        shift # Remove argument from processing
        ;;
        --no-cpu)
        RUN_CPU=false
        shift # Remove argument from processing
        ;;
        *)
        # Skip unknown options
        ;;
    esac
done

echo "Making solutions"
make > /dev/null

# Generate input if --no-gen is not specified
if [ "$GEN_INPUT" = true ]; then
    echo "Generating input"
    ./bin/generate_input input/input.txt 1000 100000
else
    echo "Skipping input generation"
fi

# Run CPU if --no-cpu is not specified
if [ "$RUN_CPU" = true ]; then
    echo "Running CPU"
    CPU_START_TIME=$(date +%s%N | cut -b1-13)
    ./bin/hamming_one_cpu input/input.txt > output/output1.txt
    CPU_END_TIME=$(date +%s%N | cut -b1-13)
else
    echo "Skipping CPU run, using GPU output as baseline"
fi

# Run GPU
echo "Running GPU"
GPU_START_TIME=$(date +%s%N | cut -b1-13)
./bin/hamming_one_gpu input/input.txt > output/output2.txt
GPU_END_TIME=$(date +%s%N | cut -b1-13)

# If CPU run was skipped, use GPU output as baseline
if [ "$RUN_CPU" = false ]; then
    cp output/output2.txt output/output1.txt
fi

# Run Optimized GPU
echo "Running optimised GPU"
OPT_GPU_START_TIME=$(date +%s%N | cut -b1-13)
./bin/hamming_one_optimised_gpu input/input.txt > output/output3.txt
OPT_GPU_END_TIME=$(date +%s%N | cut -b1-13)

# Compare results
echo "GPU: " `./bin/test_solution output/output1.txt output/output2.txt`
echo "Optimised GPU: " `./bin/test_solution output/output1.txt output/output3.txt`

# Timing output
if [ "$RUN_CPU" = true ]; then
    echo "CPU calculations took $(($CPU_END_TIME - $CPU_START_TIME)) milliseconds to complete"
fi
echo "GPU calculations took $(($GPU_END_TIME - $GPU_START_TIME)) milliseconds to complete"
echo "Optimised GPU calculations took $(($OPT_GPU_END_TIME - $OPT_GPU_START_TIME)) milliseconds to complete"