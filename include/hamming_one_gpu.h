#ifndef HAMMING_ONE_GPU_H
#define HAMMING_ONE_GPU_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda_runtime.h>
using namespace std;

/**
* @file hamming_one_gpu.h
* @brief This file contains the CUDA kernel to find pairs of sequences with a Hamming distance of one.
* Architecture: GPU
* Complexity: O(L * M^2)
*/

#define SIZE_OF_FIFO_TXT (long long int)1e15
#define BITS_IN_INT 31

/**
 * @brief Reads input data from a file and stores it in an integer array.
 * 
 * This function reads the sequence length (L), the number of sequences (M),
 * and the sequences themselves from the file. The sequences are packed into
 * integers, with each bit in an integer representing a boolean value. This
 * reduces the space requirements for the sequences.
 * 
 * @param file_path Path to the input file.
 * @param L Reference to the sequence length (will be adjusted based on packing).
 * @param M Reference to the number of sequences.
 * @param h_input Reference to the pointer where the packed sequence data will be stored.
 */
void read_input(char* file_path, int& L, int& M, int*& h_input);

/**
 * @brief CUDA kernel to find pairs of sequences with a Hamming distance of one.
 * 
 * This kernel calculates the Hamming distance between pairs of sequences. It uses
 * bitwise operations to determine the Hamming distance between sequences packed
 * into integers. If two sequences differ by exactly one bit, their indices are printed.
 * 
 * @param d_input Pointer to the device array containing the packed sequences.
 * @param L The length of each sequence (in terms of packed integers).
 * @param M The number of sequences.
 */
__global__ void find_hamming_one(int* d_input, int L, int M);

#endif // HAMMING_ONE_GPU_H