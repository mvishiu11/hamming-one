// Creates tests for Hamming One problem
//
// In order to maximise amount of comparisions required to 
// calculate result by brute force solutions, sequences 
// often differ on the very last positions
//

#include "generate_input.h"

#define ERR(source) (fprintf(stderr,"%s:%d\n",__FILE__,__LINE__),\
                     perror(source),\
                     exit(EXIT_FAILURE))

void usage(char *name){
    fprintf(stderr,"USAGE: %s <output_file_path> [L] [M]\n",name);
    fprintf(stderr,"L - sequence lenght [OPTIONAL, L = %d], L >= %d\n", DEFAULT_L_VALUE, MIN_L_VALUE);
    fprintf(stderr,"M - number of sequences [OPTIONAL, M = %d], M >= %d\n", DEFAULT_M_VALUE, MIN_M_VALUE);
    exit(EXIT_FAILURE);
}

int _ceil(double variable) {
    int new_variable = (int)variable;
    if ((double)new_variable == variable) return new_variable;
    else return new_variable + 1;
}

void write_data_to_file(char * file_path, bool * output, int L, int M) {
    ofstream fileStream;
    fileStream.open(file_path, ios::out | ios::trunc);
    if (!fileStream.is_open()) ERR("ostream.open");

    fileStream << L << " " << M << "\n";

    for (int i = 0; i < M; i++) {
        for (int o = 0; o < L; o++) {
            fileStream << output[o + i * L] << " ";
        }
        fileStream << "\n";
    }
    fileStream.close();
}

void __global__ generate_data(bool * output, int L, int M) {
    int index = blockIdx.x * 1024 + threadIdx.x;
    if (index >= M || index < M / 10) return;

    curandState_t state;
    curand_init(clock64(), 0, 0, &state);

    int sourceIndex = curand(&state) % (M / 10);
    
    for (int i = 0; i < L; i++) {
        output[i + index * L] = output[i + sourceIndex * L];
    }

    int numOfChanges = curand(&state) % 4;
    for (int o = 0; o < numOfChanges; o++) {
        int pos = curand(&state) % (L / 100 * 5) + (L / 100 * 95);
        output[pos + index * L] = 1 - output[pos + index * L];
    }
}

void fill_first_sequences(bool* output, int L, int prefix) {
    srand(time(NULL));
    for (int i = 0; i < 3 * L * prefix; i++) {
        output[i] = rand() % 2;
    }
}

int main(int argc, char ** argv) {
    int L, M;
    L = DEFAULT_L_VALUE;
    M = DEFAULT_M_VALUE;

    if (argc > 4 || argc < 2) usage(argv[0]);
    if (argc == 4) M = atoi(argv[3]);
    if (argc >= 3) L = atoi(argv[2]);
    if (L < MIN_L_VALUE || M < MIN_M_VALUE) usage(argv[0]);

    bool * output;
    cudaMallocManaged(&output, sizeof(bool) * L * M);
    if (output == NULL) ERR("cudaMallocManaged");

    fill_first_sequences(output, L, M / 10);

    int threads, blocks;
    threads = 1024;
    blocks = _ceil((double)M / threads);

    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, SIZE_OF_FIFO_TXT);
    generate_data<<<blocks, threads>>>(output, L, M);
    cudaDeviceSynchronize();

    write_data_to_file(argv[1], output, L, M);
    cudaFree(output);
}