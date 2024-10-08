#include "hamming_one_hashtable_gpu.h"

#define ERR(source) (fprintf(stderr,"%s:%d\n",__FILE__,__LINE__),\
                     perror(source),\
                     exit(EXIT_FAILURE))

void usage(char *name){
    fprintf(stderr,"USAGE: %s <input_file_path>\n",name);
    exit(EXIT_FAILURE);
}

__host__ __device__ bool Triplet::operator<(const Triplet& triplet) const {
    if (hash1 < triplet.hash1) return true;
    if (hash1 == triplet.hash1 && hash2 < triplet.hash2) return true;
    return false;
}

int _ceil(double variable) {
    int new_variable = (int)variable;
    if ((double)new_variable == variable) return new_variable;
    else return new_variable + 1;
}

__global__ void hash_table_init(Pair* hash_table, int table_size) {
    int index = blockIdx.x * 1024 + threadIdx.x;
    if (index >= table_size) return;
    hash_table[index].hash = EMPTY_KEY;
}

__device__ void hash_table_insert(Pair* hash_table, long long int hash1, long long int hash2, int index, int table_size) {
    unsigned long long pos = hash1 % table_size;
    int quad = 1;
    while (true) {
        int old = atomicCAS((unsigned long long int*)&hash_table[pos].hash, EMPTY_KEY, (unsigned long long int)hash2);
        if (old == EMPTY_KEY || old == hash2) {
            hash_table[pos].index = index;
            break;
        }
        pos = (pos + quad * quad) % table_size;
        quad++;
    }
}

__device__ int hash_table_lookup(Pair* hash_table, long long int hash1, long long int hash2, int table_size) {
    unsigned long long pos = hash1 % table_size;
    int quad = 1;
    while (true) {
        if (hash_table[pos].hash == hash2) {
            return hash_table[pos].index;
        }
        if (hash_table[pos].hash == EMPTY_KEY) {
            return -1;
        }
        pos = (pos + quad * quad) % table_size;
        quad++;
    }
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

__global__ void calculate_hashes(Triplet* d_hashes_map, Pair* d_hash_table, bool *d_input, int L, int M, int table_size) {

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
    hash_table_insert(d_hash_table, hash1, hash2, index, table_size);
    triplet.hash1 = hash1;
    triplet.hash2 = hash2;
    triplet.index = index;
    d_hashes_map[index] = triplet;
}

__global__ void find_hamming_one(Triplet* d_hashes_map, Pair* d_hash_table, bool* d_input, int L, int M, int table_size) {

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
        int original_o = hash_table_lookup(d_hash_table, temp_hash1, temp_hash2, table_size);
        int o = original_o;
        while(o != -1 && o < M && d_hashes_map[o].hash1 == temp_hash1 && d_hashes_map[o].hash2 == temp_hash2) {
            if (d_hashes_map[index].index < d_hashes_map[o].index) 
                printf("%d %d\n", d_hashes_map[index].index, d_hashes_map[o].index);
            o++;
        }
        o = original_o - 1;
        while(o != -1 && d_hashes_map[o].hash1 == temp_hash1 && d_hashes_map[o].hash2 == temp_hash2) {
            if (d_hashes_map[index].index < d_hashes_map[o].index) 
                printf("%d %d\n", d_hashes_map[index].index, d_hashes_map[o].index);
            o--;
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
    Pair* d_hash_table;
    int table_size = 4 * M;
    cudaMalloc(&d_hashes_map, sizeof(Triplet) * M);
    cudaMalloc(&d_hash_table, sizeof(Pair) * table_size);

    hash_table_init<<<blocks, threads>>>(d_hash_table, table_size);
    calculate_hashes<<<blocks, threads>>>(d_hashes_map, d_hash_table, d_input, L, M, table_size);
    
    thrust::sort(thrust::device, d_hashes_map, d_hashes_map + M);
    
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, SIZE_OF_FIFO_TXT);
    find_hamming_one<<<blocks, threads>>>(d_hashes_map, d_hash_table, d_input, L, M, table_size);
    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_hashes_map);
    return EXIT_SUCCESS;
}