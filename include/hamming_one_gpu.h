#ifndef HAMMING_ONE_GPU_H
#define HAMMING_ONE_GPU_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda_runtime.h>
using namespace std;

#define SIZE_OF_FIFO_TXT (long long int)1e15
#define BITS_IN_INT 31

void read_input(char* file_path, int& L, int& M, int*& h_input);

__global__ void find_hamming_one(int* d_input, int L, int M);

#endif