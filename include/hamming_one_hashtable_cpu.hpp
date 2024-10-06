#ifndef HAMMING_ONE_HASHTABLE_CPU_HPP
#define HAMMING_ONE_HASHTABLE_CPU_HPP

#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <unordered_map>
using namespace std;

/**
 * @file hamming_one_cpu.hpp
 * @brief This file contains the CPU implementation to find pairs of sequences with a Hamming distance of one,
 *        using unordered maps and rolling polynomial hashing to reduce complexity to O(M * L).
 * Architecture: CPU
 * Complexity: O(M * L)
 */

#define MOD1 100000004917
#define MOD2 99999981101
#define P1 29
#define P2 41

/**
 * @brief Reads input data from a file and initializes input sequences.
 * 
 * This function reads the sequence length (L), the number of sequences (M),
 * and the sequences themselves from a file. The data is stored in the
 * dynamically allocated array `h_input`.
 * 
 * @param file_path Path to the input file.
 * @param L Reference to the sequence length (to be read from the file).
 * @param M Reference to the number of sequences (to be read from the file).
 * @param h_input Reference to the pointer where the sequence data will be stored.
 */
void read_input(char* file_path, int& L, int& M, bool*& h_input);

/**
 * @brief Calculates hash values for the input sequences.
 * 
 * This function computes two hash values (modular hash values with different
 * primes) for each sequence. These hash values are stored in the arrays
 * `d_hashes1` and `d_hashes2`.
 * 
 * @param d_hashes1 Pointer to the array where the first set of hashes will be stored.
 * @param d_hashes2 Pointer to the array where the second set of hashes will be stored.
 * @param d_input Pointer to the input boolean array representing the sequences.
 * @param L The length of each sequence.
 * @param M The number of sequences.
 */
void calculate_hashes(long long int *d_hashes1, long long int *d_hashes2, bool *d_input, int L, int M);

/**
 * @brief Finds sequences that have a Hamming distance of one.
 * 
 * This function finds pairs of sequences with a Hamming distance of one using
 * the precomputed hash values and unordered maps for faster lookups. It outputs
 * the indices of sequence pairs that have exactly one bit difference.
 * 
 * Architecture: CPU
 * Complexity: O(M * L)
 * 
 * @param h_hashes1 Pointer to the first set of hash values for the sequences.
 * @param h_hashes2 Pointer to the second set of hash values for the sequences.
 * @param L The length of each sequence.
 * @param M The number of sequences.
 */
void find_hamming_one(long long int *h_hashes1, long long int *h_hashes2, int L, int M);

#endif // HAMMING_ONE_HASHTABLE_CPU_HPP
