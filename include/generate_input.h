#ifndef GENERATE_INPUT_H
#define GENERATE_INPUT_H

#include <iostream>
#include <fstream>
#include <curand.h>
#include <curand_kernel.h>
using namespace std;

#define DEFAULT_L_VALUE 1000
#define DEFAULT_M_VALUE (int)1e5
#define SIZE_OF_FIFO_TXT (long long int)1e15
#define MIN_L_VALUE 3
#define MIN_M_VALUE 20

/**
 * @brief Writes generated data to a specified file.
 * 
 * This function opens a file, writes the sequence length (L) and the number
 * of sequences (M) on the first line, followed by the generated boolean data
 * for each sequence. Each sequence is written as a line of 1's and 0's.
 * 
 * @param file_path The file path where the output will be written.
 * @param output Pointer to the array containing the generated boolean data.
 * @param L The length of each sequence.
 * @param M The number of sequences.
 */
void write_data_to_file(char *file_path, bool *output, int L, int M);

/**
 * @brief Generates boolean data sequences using CUDA.
 * 
 * This CUDA kernel generates boolean sequences, ensuring sequences are similar
 * except for random changes in the last few positions. It aims to maximize the
 * number of comparisons required to calculate results in brute force solutions 
 * by ensuring that the sequences are similar and often differ only in the last
 * few positions.
 * 
 * @param output Pointer to the array where generated boolean data will be stored.
 * @param L The length of each sequence.
 * @param M The number of sequences.
 */
void __global__ generate_data(bool *output, int L, int M);

#endif // GENERATE_INPUT_H