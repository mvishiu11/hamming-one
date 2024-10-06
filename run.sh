echo "Making solutions"
make > /dev/null

echo "Generating input"
./bin/generate_input input/input.txt 1000 100000

echo "Running CPU"
CPU_START_TIME=$(date +%s%N | cut -b1-13)
./bin/hamming_one_cpu input/input.txt > output/output1.txt
CPU_END_TIME=$(date +%s%N | cut -b1-13)

echo "Running GPU"
GPU_START_TIME=$(date +%s%N | cut -b1-13)
./bin/hamming_one_gpu input/input.txt > output/output2.txt
GPU_END_TIME=$(date +%s%N | cut -b1-13)

echo "Running optimised GPU"
OPT_GPU_START_TIME=$(date +%s%N | cut -b1-13)
./bin/hamming_one_optimised_gpu input/input.txt > output/output3.txt
OPT_GPU_END_TIME=$(date +%s%N | cut -b1-13)


echo "GPU: " `./bin/test_solution output/output1.txt output/output2.txt` 
echo "Optimised GPU: "`./bin/test_solution output/output1.txt output/output3.txt` 

echo "CPU calculations took $(($CPU_END_TIME - $CPU_START_TIME)) miliseconds to complete"
echo "GPU calculations took $(($GPU_END_TIME - $GPU_START_TIME)) miliseconds to complete"
echo "Optimised GPU calculations took $(($OPT_GPU_END_TIME - $OPT_GPU_START_TIME)) miliseconds to complete"