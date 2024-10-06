// Hamming One
// Architecture: GPU
// Complexity: L * MlogM
//
// Complexity can be easily lowered to O(L * M) 
// by using a dictonary that supports insert and 
// chech_if_consists both in o(1) complexity
//

#include "hamming_one_optimised_gpu.h"

#define ERR(source) (fprintf(stderr,"%s:%d\n",__FILE__,__LINE__),\
                     perror(source),\
                     exit(EXIT_FAILURE))

void usage(char *name){
    fprintf(stderr,"USAGE: %s <input_file_path>\n",name);
    exit(EXIT_FAILURE);
}

int _ceil(double variable) {
    int new_variable = (int)variable;
    if ((double)new_variable == variable) return new_variable;
    else return new_variable + 1;
}

void read_input(char* file_path, int& L, int& M, bool*& h_input) {
    ifstream fileStream;    
    fileStream.open(file_path, ios::in);
    if (!fileStream.is_open()) ERR("ifstream.open");
    fileStream >> L >> M;

    h_input = new bool[L * M];
    if (h_input == NULL) ERR("operator new");

    for (int i = 0; i < M; i++) {
        for (int o = 0; o < L; o++) {
            fileStream >> h_input[o + i * L];
        }
    }
}

__device__ int binsearch(Triplet* d_hashes_map, int M, long long int hash1, long long int hash2) {
    int p = 0;
    int k = M;
    int sr;
    while (p < k) {
        sr = p + ((k - p) >> 2);
        if (hash1 < d_hashes_map[sr].hash1) k = sr;
        else if (hash1 == d_hashes_map[sr].hash1 && hash2 <= d_hashes_map[sr].hash2) k = sr;
        else p = sr + 1;
    }
    if (sr < M && d_hashes_map[sr].hash1 < hash1) sr++;
    else if (sr < M && d_hashes_map[sr].hash1 == hash1 && d_hashes_map[sr].hash2 < hash2) sr++;

    if (sr == M || d_hashes_map[sr].hash1 != hash1 || d_hashes_map[sr].hash2 != hash2) return -1;
    return sr;
}

__global__ void calculate_hashes(Triplet* d_hashes_map, bool *d_input, int L, int M) {

    int index = blockIdx.x * 1024 + threadIdx.x;
    if (index >= M) return;

    long long int hash1 = 0, hash2 = 0;
    long long int p1 = P1;
    long long int p2 = P2;
    d_input += index * L;
    for (int i = 0; i < L; i++) {
        hash1 = (hash1 + p1 * d_input[i]) % MOD1;
        hash2 = (hash2 + p2 * d_input[i]) % MOD2;
        p1 = (p1 * P1) % MOD1;
        p2 = (p2 * P2) % MOD2;
    }

    Triplet triplet;
    triplet.hash1 = hash1;
    triplet.hash2 = hash2;
    triplet.index = index;
    d_hashes_map[index] = triplet;
}

__global__ void find_hamming_one(Triplet* d_hashes_map, bool* d_input, int L, int M) {

    int index = blockIdx.x * 1024 + threadIdx.x;
    if (index >= M) return;
    
    int reversed_bit; // turns (1 to -1) and (0 to 1)
    long long int p1 = P1;
    long long int p2 = P2;

    long long int hash1 = d_hashes_map[index].hash1;
    long long int hash2 = d_hashes_map[index].hash2;

    long long int temp_hash1, temp_hash2;
    d_input += L * d_hashes_map[index].index;

    for (int i = 0; i < L; i++) {
        reversed_bit = 1 - 2 * d_input[i];

        temp_hash1 = (hash1 + reversed_bit * p1 + MOD1) % MOD1;
        temp_hash2 = (hash2 + reversed_bit * p2 + MOD2) % MOD2;

        int o = binsearch(d_hashes_map, M, temp_hash1, temp_hash2);
        while(o != -1 && o < M && d_hashes_map[o].hash1 == temp_hash1 && d_hashes_map[o].hash2 == temp_hash2) {
            if (d_hashes_map[index].index < d_hashes_map[o].index) 
                printf("%d %d\n", d_hashes_map[index].index, d_hashes_map[o].index);
            o++;
        }

        p1 = (p1 * P1) % MOD1;
        p2 = (p2 * P2) % MOD2;
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) usage(argv[0]);
    int L, M;
    bool *h_input, *d_input;
    read_input(argv[1], L, M, h_input);
    cudaMalloc(&d_input, L * M * sizeof(bool));
    cudaMemcpy(d_input, h_input, L * M * sizeof(bool), cudaMemcpyHostToDevice);
    delete[] h_input;
    
    int threads, blocks;
    threads = 1024;
    blocks = _ceil((double)M / threads);

    Triplet* d_hashes_map;
    cudaMalloc(&d_hashes_map, sizeof(Triplet) * M);
    calculate_hashes<<<blocks, threads>>>(d_hashes_map, d_input, L, M);
    
    thrust::sort(thrust::device, d_hashes_map, d_hashes_map + M);
    
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, SIZE_OF_FIFO_TXT);
    find_hamming_one<<<blocks, threads>>>(d_hashes_map, d_input, L, M);
    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_hashes_map);
    return EXIT_SUCCESS;
}