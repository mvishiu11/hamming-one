#ifndef HAMMING_ONE_OPTIMISED_GPU_H
#define HAMMING_ONE_OPTIMISED_GPU_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
using namespace std;

#define SIZE_OF_FIFO_TXT (long long int)1e15
#define MOD1 100000004917
#define MOD2 99999981101
#define P1 29
#define P2 41

struct Triplet {
    long long int hash1, hash2;
    int index;

    __host__ __device__ bool operator<(const Triplet& triplet) const {
        if (hash1 != triplet.hash1) return hash1 < triplet.hash1;
        if (hash2 != triplet.hash2) return hash2 < triplet.hash2;
        return index < triplet.index;
    }
};

void read_input(char* file_path, int& L, int& M, bool*& h_input);

__device__ int binsearch(Triplet* d_hashes_map, int M, long long int hash1, long long int hash2);

__global__ void calculate_hashes(Triplet* d_hashes_map, bool *d_input, int L, int M);

__global__ void find_hamming_one(Triplet* d_hashes_map, bool* d_input, int L, int M);

#endif