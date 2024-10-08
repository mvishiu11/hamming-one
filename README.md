# Hamming One Algorithm Project

![CUDA](https://img.shields.io/badge/CUDA-green?logo=nvidia)
![C++](https://img.shields.io/badge/language-C++-blue)
![GPU](https://img.shields.io/badge/GPU-Supported-orange)

## Table of Contents
- [Hamming One Algorithm Project](#hamming-one-algorithm-project)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Theory](#theory)
  - [Basic Algorithm Explanation](#basic-algorithm-explanation)
    - [Steps:](#steps)
    - [Complexity Analysis:](#complexity-analysis)
    - [Correctness:](#correctness)
    - [Optimizations to this algorithm that were shown in this project:](#optimizations-to-this-algorithm-that-were-shown-in-this-project)
  - [Optimized Algorithm Explanation](#optimized-algorithm-explanation)
    - [Steps:](#steps-1)
    - [Complexity Analysis:](#complexity-analysis-1)
    - [Correctness](#correctness-1)
    - [Optimizations to this algorithm that were shown in this project:](#optimizations-to-this-algorithm-that-were-shown-in-this-project-1)
  - [Assumptions](#assumptions)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Running the Algorithms](#running-the-algorithms)
    - [Running Tests](#running-tests)
  - [Benchmarks](#benchmarks)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction

This project implements various versions of the Hamming One algorithm, which is used to find all pairs of strings that have a Hamming distance of one. The project includes CPU, GPU, and optimized GPU implementations, as well as versions using hash tables for improved performance.

## Theory

The Hamming distance between two strings of equal length is the number of positions at which the corresponding symbols differ. For example, the Hamming distance between "karolin" and "kathrin" is 3.

The Hamming One algorithm specifically looks for pairs of strings that have a Hamming distance of exactly one. This is useful in various applications, including error detection and correction, bioinformatics, and data clustering.

## Basic Algorithm Explanation

The basic idea of the Hamming One algorithm is to iterate through each pair of strings and count the number of positions at which they differ. If this count is exactly one, the pair is considered a match.

### Steps:
1. **Input Reading**: Read the input strings from a file.
2. **Pairwise Comparison**: Compare each string with every other string.
3. **Hamming Distance Calculation**: For each pair, calculate the Hamming distance.
4. **Result Storage**: Store the pairs with a Hamming distance of one.

### Complexity Analysis:
- **Time Complexity**: O(n^2 * l), where n is the number of strings and l is the length of each string.
- **Space Complexity**: O(n), where n is the number of strings.

### Correctness:
The basic algorithm is deterministic and correct, as it directly compares each pair of strings and calculates the Hamming distance. However, it may not be efficient for large datasets due to its quadratic time complexity.

### Optimizations to this algorithm that were shown in this project:
- **GPU Acceleration**: Use GPU to parallelize the comparisons and distance calculations.

## Optimized Algorithm Explanation

This algorithm identifies all pairs of binary sequences with a Hamming distance of exactly one. Given \( n \) binary sequences, each of length \( l \), the algorithm efficiently finds pairs of sequences where exactly one bit differs between them.

### Steps:

1. **Hashing the Sequences**: 
   - Two hash values are computed for each sequence using prime constants \( P_1 \), \( P_2 \), and large modulus values \( M_1 \) and \( M_2 \).
   - The hash functions are as follows:
     \[
     \text{hash1}_i = \left( \sum_{j=0}^{l-1} P_1^j \cdot a_i[j] \right) \mod M_1
     \]
     \[
     \text{hash2}_i = \left( \sum_{j=0}^{l-1} P_2^j \cdot a_i[j] \right) \mod M_2
     \]
   - These calculations avoid exceeding numeric limits by using modular arithmetic.

2. **Storing and Sorting Hashes**: 
   - The hash pairs \( (\text{hash1}_i, \text{hash2}_i, i) \) for each sequence \( i \) are stored in an array.
   - This array is then sorted. Sorting helps in the efficient lookup of sequences that differ by exactly one bit.

3. **Identifying Pairs with Hamming Distance of One**:
   - For two sequences \( a_i \) and \( a_j \) with a Hamming distance of 1, there exists a single differing bit at position \( k \).
   - By computing new hashes for \( a_i \) with the \( k \)-th bit toggled, the algorithm checks for matches in the sorted array using a binary search (lower bound algorithm).
   - This approach enables each comparison to run in \( O(\log n) \), and with \( l \) possible bit positions to toggle, the time complexity for this step is \( O(nl \log n) \).

### Complexity Analysis:

The overall complexity of the algorithm is:
- **Hash Calculation**: \( O(nl) \) as it processes \( n \) sequences of length \( l \).
- **Sorting**: \( O(n \log n) \).
- **Searching**: \( O(nl \log n) \), as for each of the \( n \) sequences, \( l \) positions are considered, and each lookup takes \( O(\log n) \).

Thus, the total complexity is \( O(nl \log n) \).

### Correctness

The algorithm is probabilistic, given the possibility of hash collisions where different sequences might share identical hash values. However, with the chosen modulus values, the probability of a collision is extremely low (approximately \( 10^{-17} \)), making the algorithm reliable in nearly all practical cases.

### Optimizations to this algorithm that were shown in this project:
- **GPU Acceleration**: Use GPU to parallelize the hashing, sorting, and searching steps.
- **Hash Table [IN IMPLEMENTATION]**: Use a hash table to store the hash pairs for efficient lookups - this reduces the time complexity of the search step to \( O(nl) \), making the overall complexity \( O(nl) \).

## Assumptions

- All input strings are of equal length.
- The input file is properly formatted and contains one string per line.
- The system has the necessary hardware and software to support GPU acceleration if using GPU-based implementations.

## Setup

### Prerequisites

- C++ Compiler
- Make
- CUDA Toolkit (for GPU implementations)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/hamming-one-algorithm.git
    cd hamming-one-algorithm
    ```

2. Build the project:
    ```sh
    make all
    ```

## Usage

### Running the Algorithms

The simplest way to run the algos is to use the provided `run.sh` script. This script will generate input, run all the algorithms, and compare the results.

```sh
./run.sh
```

Alternatively, you can run each algorithm individually:

- **Generate Input**:
    ```sh
    ./bin/generate_input
    ```

- **CPU Implementation**:
    ```sh
    ./bin/hamming_one_cpu
    ```

- **GPU Implementation**:
    ```sh
    ./bin/hamming_one_gpu
    ```

- **Optimized GPU Implementation**:
    ```sh
    ./bin/hamming_one_optimised_gpu
    ```

- **Hash Table CPU Implementation**:
    ```sh
    ./bin/hamming_one_hashtable_cpu
    ```

- **Hash Table GPU Implementation**:
    ```sh
    ./bin/hamming_one_hashtable_gpu
    ```

### Running Tests

To test the solution:
```sh
./bin/test_solution
```

## Benchmarks

Benchmark results can be found in the [benchmarks/timing_results.txt](benchmarks/timing_results.txt) file. These results compare the performance of different implementations.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.