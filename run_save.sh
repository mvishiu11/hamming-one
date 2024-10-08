#!/bin/bash

# Default flags
GEN_INPUT=true
RUN_CPU=true
RUN_GPU_HASH=false

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
        --with-gpu-hash)
        RUN_GPU_HASH=true
        shift # Remove argument from processing
        ;;
        *)
        # Skip unknown options
        ;;
    esac
done

echo "Making solutions"
make all > /dev/null

# Define an array of sample sizes
SAMPLE_SIZES=(5000 10000 100000 200000 300000)  # Modify as needed
SEQ_LENGTH=10000
RESULTS_FILE="benchmarks/timing_results.txt"

# Initialize the results file
echo "Sample Size,Sequence length,CPU Time (ms),CPU Hashtable Time (ms),GPU Time (ms),Optimized GPU Time (ms),GPU Hashtable Time (ms)" > $RESULTS_FILE

# Loop over each sample size
for SAMPLE_SIZE in "${SAMPLE_SIZES[@]}"
do
    echo "Running tests with sample size: $SAMPLE_SIZE"

    # Generate input if --no-gen is not specified
    if [ "$GEN_INPUT" = true ]; then
        echo "Generating input"
        ./bin/generate_input input/input.txt "$SEQ_LENGTH" "$SAMPLE_SIZE"
    else
        echo "Skipping input generation"
    fi

    # Initialize timing variables
    CPU_TIME=0
    CPU_HASHTABLE_TIME=0
    GPU_TIME=0
    OPT_GPU_TIME=0
    HASH_GPU_TIME=0

    # Run CPU if --no-cpu is not specified
    if [ "$RUN_CPU" = true ]; then
        echo "Running CPU"
        CPU_START_TIME=$(date +%s%N | cut -b1-13)
        ./bin/hamming_one_cpu input/input.txt > output/output1.txt
        CPU_END_TIME=$(date +%s%N | cut -b1-13)
        CPU_TIME=$(($CPU_END_TIME - $CPU_START_TIME))

        echo "Running CPU with hashtable"
        CPU_HASHTABLE_START_TIME=$(date +%s%N | cut -b1-13)
        ./bin/hamming_one_hashtable_cpu input/input.txt > output/output4.txt
        CPU_HASHTABLE_END_TIME=$(date +%s%N | cut -b1-13)
        CPU_HASHTABLE_TIME=$(($CPU_HASHTABLE_END_TIME - $CPU_HASHTABLE_START_TIME))
    else
        echo "Skipping CPU run, using GPU output as baseline"
    fi

    # Run GPU
    echo "Running GPU"
    GPU_START_TIME=$(date +%s%N | cut -b1-13)
    ./bin/hamming_one_gpu input/input.txt > output/output2.txt
    GPU_END_TIME=$(date +%s%N | cut -b1-13)
    GPU_TIME=$(($GPU_END_TIME - $GPU_START_TIME))

    # If CPU run was skipped, use GPU output as baseline
    if [ "$RUN_CPU" = false ]; then
        cp output/output2.txt output/output1.txt
    fi

    # Run Optimized GPU
    echo "Running optimised GPU"
    OPT_GPU_START_TIME=$(date +%s%N | cut -b1-13)
    ./bin/hamming_one_optimised_gpu input/input.txt > output/output3.txt
    OPT_GPU_END_TIME=$(date +%s%N | cut -b1-13)
    OPT_GPU_TIME=$(($OPT_GPU_END_TIME - $OPT_GPU_START_TIME))

    # Run HashTable GPU
    if [ "$RUN_GPU_HASH" = true ]; then
        echo "Running GPU with hashtable"
        HASH_GPU_START_TIME=$(date +%s%N | cut -b1-13)
        ./bin/hamming_one_hashtable_gpu input/input.txt > output/output5.txt
        HASH_GPU_END_TIME=$(date +%s%N | cut -b1-13)
        HASH_GPU_TIME=$(($HASH_GPU_END_TIME - $HASH_GPU_START_TIME))
    fi

    # Save results to the text file
    if [ "$RUN_CPU" = true ]; then
        echo "$SAMPLE_SIZE,$SEQ_LENGTH,$CPU_TIME,$CPU_HASHTABLE_TIME,$GPU_TIME,$OPT_GPU_TIME,$HASH_GPU_TIME" >> $RESULTS_FILE
    else
        echo "$SAMPLE_SIZE,$SEQ_LENGTH,$CPU_TIME,$CPU_HASHTABLE_TIME,$GPU_TIME,$OPT_GPU_TIME" >> $RESULTS_FILE
    fi

    # Output results to the terminal
    echo "Results for sample size $SAMPLE_SIZE:"
    if [ "$RUN_CPU" = true ]; then
        echo "CPU calculations took $CPU_TIME milliseconds to complete"
        echo "CPU with hashtable calculations took $CPU_HASHTABLE_TIME milliseconds to complete"
    fi
    echo "GPU calculations took $GPU_TIME milliseconds to complete"
    echo "Optimised GPU calculations took $OPT_GPU_TIME milliseconds to complete"
    if [ "$RUN_GPU_HASH" = true ]; then
        echo "GPU with hashtable calculations took $HASH_GPU_TIME milliseconds to complete"
    fi

done

echo "All tests completed. Results saved in $RESULTS_FILE."
