#ifndef HAMMING_ONE_HASHTABLE_GPU_H
#define HAMMING_ONE_HASHTABLE_GPU_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <unordered_map>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
using namespace std;

/**
 * @file hamming_one_optimised_gpu.h
 * @brief This file contains the optimised CUDA kernel to find pairs of sequences with a Hamming distance of one,
 *        using rolling polynomial hashing to reduce the number of comparisons and hash table to reduce time complexity 
 *        of search.
 * Architecture: GPU
 * Complexity: O(L * M)
 */

#define SIZE_OF_FIFO_TXT (long long int)1e15
#define MOD1 100000004917
#define MOD2 99999981101
#define P1 29
#define P2 41
#define EMPTY_KEY 0xffffffffffffffff
#define EMPTY_INDEX -1

/**
 * @brief Structure representing the hashes and index of each sequence.
 * 
 * This structure stores two hash values (`hash1` and `hash2`) and the index
 * of the sequence. It provides a comparison operator (`<`) to allow sorting
 * of Triplets based on their hash values.
 */
struct Triplet {
    long long int hash1, hash2;
    int index;

    /**
     * @brief Comparison operator for sorting Triplets.
     * 
     * Compares two Triplets first by `hash1`, then by `hash2`, and finally by
     * their `index` if the hash values are equal.
     * 
     * @param triplet The other Triplet to compare with.
     * @return True if the current Triplet is less than the other Triplet.
     */
    __host__ __device__ bool operator<(const Triplet& triplet) const;
};

/**
 * @brief Structure representing the an index and hash value pair.
 * 
 * This structure stores an index and a hash value and is meant for storage in hash maps.
 */
struct Pair {
    int index;
    long long int hash;
};

/**
 * @brief Reads input data from a file and stores it in a boolean array.
 * 
 * This function reads the sequence length (L), the number of sequences (M),
 * and the sequences themselves from the file, storing them in the array
 * `h_input`.
 * 
 * @param file_path The path to the input file.
 * @param L Reference to the sequence length.
 * @param M Reference to the number of sequences.
 * @param h_input Reference to the pointer where input data will be stored.
 */
void read_input(char* file_path, int& L, int& M, bool*& h_input);

/**
 * @brief Performs a binary search on the hash map to find a matching hash.
 * 
 * This device function searches the sorted `d_hashes_map` for a sequence
 * with the given hash values (`hash1` and `hash2`). It returns the index of
 * the matching sequence or -1 if no match is found.
 * 
 * @param d_hashes_map Pointer to the device array of Triplet structures.
 * @param M The number of sequences.
 * @param hash1 The first hash value to search for.
 * @param hash2 The second hash value to search for.
 * @return The index of the matching sequence or -1 if not found.
 */
__device__ int binsearch(Triplet* d_hashes_map, int M, long long int hash1, long long int hash2);

/**
 * @brief CUDA kernel to calculate hashes for the input sequences.
 * 
 * This kernel computes two hash values for each sequence in the input
 * data and stores them in the `d_hashes_map` array.
 * 
 * @param d_hashes_map Device array where the computed Triplet hashes will be stored.
 * @param d_input Device array of input boolean sequences.
 * @param L The length of each sequence.
 * @param M The number of sequences.
 */
__global__ void calculate_hashes(Triplet* d_hashes_map, Triplet* d_hash_table, bool *d_input, int L, int M);

/**
 * @brief CUDA kernel to find sequences with a Hamming distance of one.
 * 
 * This kernel searches for pairs of sequences that differ by exactly one bit
 * using the precomputed hash values. The results (indices of matching pairs)
 * are printed to the console.
 * 
 * @param d_hashes_map Device array of sorted Triplet hashes.
 * @param d_input Device array of input boolean sequences.
 * @param L The length of each sequence.
 * @param M The number of sequences.
 */
__global__ void find_hamming_one(Triplet* d_hashes_map, Triplet* d_hash_table, bool* d_input, int L, int M);

#endif // HAMMING_ONE_HASHTABLE_GPU_H