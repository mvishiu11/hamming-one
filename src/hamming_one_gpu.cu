// Hamming One
// Architecture: GPU
// Complexity: L * M^2
//

#include <iostream>
#include <fstream>
#include <cstring>
using namespace std;

#define SIZE_OF_FIFO_TXT (long long int)1e15
#define BITS_IN_INT 31
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

void read_input(char* file_path, int& L, int& M, int*& h_input) {
    ifstream fileStream;    
    fileStream.open(file_path, ios::in);
    if (!fileStream.is_open()) ERR("ifstream.open");
    fileStream >> L >> M;

    int newL = _ceil((double)L / BITS_IN_INT);

    h_input = new int[newL * M];
    if (h_input == NULL) ERR("operator new");
    memset(h_input, 0, sizeof(int) * newL * M);

    int current_bit;
    for (int i = 0; i < M; i++) {
        for (int o = 0; o < newL; o++) {
            for (int j = 0; j < BITS_IN_INT && o * BITS_IN_INT + j < L; j++) {
                fileStream >> current_bit;
                h_input[o + i * newL] = (h_input[o + i * newL] << 1) + current_bit;
            }
        }
    }
    L = newL;
}

__global__ void find_hamming_one(int* d_input, int L, int M) {
    int index = blockIdx.x * 1024 + threadIdx.x;
    int hamming_distance;
    for (int i = index + 1; i < M; i++) {
        hamming_distance = 0;

        for (int o = 0; o < L && hamming_distance <= 1; o++) {
            int num = d_input[o + index * L] ^ d_input[o + i * L];
            if (num != 0) {
                if ((num & (num - 1)) == 0) hamming_distance++; // difference is a power of 2
                else hamming_distance += 2; // else
            }
        }
        if (hamming_distance == 1) printf("%d %d\n", i, index);
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) usage(argv[0]);
    int L, M;
    int *h_input, *d_input;
    read_input(argv[1], L, M, h_input);

    cudaMalloc(&d_input, L * M * sizeof(int));
    cudaMemcpy(d_input, h_input, L * M * sizeof(int), cudaMemcpyHostToDevice);
    delete[] h_input;

    int threads, blocks;
    threads = 1024;
    blocks = _ceil((double)M / threads);


    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, SIZE_OF_FIFO_TXT);
    find_hamming_one<<<blocks, threads>>>(d_input, L, M);
    cudaDeviceSynchronize();

    cudaFree(d_input);
    return EXIT_SUCCESS;
}