#include "hamming_one_cpu.hpp"

#define ERR(source) (fprintf(stderr,"%s:%d\n",__FILE__,__LINE__),\
                     perror(source),\
                     exit(EXIT_FAILURE))

void usage(char *name){
    fprintf(stderr,"USAGE: %s <input_file_path>\n",name);
    exit(EXIT_FAILURE);
}

void read_input(char* file_path, int& L, int& M, bool*& h_input) {
    ifstream fileStream;    
    fileStream.open(file_path, ios::in);
    if (!fileStream.is_open()) ERR("ifstream.open");
    fileStream >> L >> M;

    h_input = new bool[L * M];
    if (h_input == nullptr) ERR("operator new");

    for (int i = 0; i < M; i++) {
        for (int o = 0; o < L; o++) {
            fileStream >> h_input[o + i * L];
        }
    }
}

void calculate_hashes(long long int *d_hashes1, long long int *d_hashes2, bool *d_input, int L, int M) {
    long long int p1, p2;
    for (int index = 0; index < M; index++) {
        p1 = P1;
        p2 = P2;
        for (int i = 0; i < L; i++) {
            d_hashes1[index] = (d_hashes1[index] + p1 * d_input[i + index * L]) % MOD1;
            d_hashes2[index] = (d_hashes2[index] + p2 * d_input[i + index * L]) % MOD2;
            p1 = (p1 * P1) % MOD1;
            p2 = (p2 * P2) % MOD2;
        }
    }
}

void find_hamming_one(long long int *h_hashes1, long long int *h_hashes2, int L, int M) {
    map<pair<long long int, long long int>, vector<int>> hash_map;

    long long int p1, p2;
    for (int i = M - 1; i >= 0; i--) {
        p1 = P1;
        p2 = P2;
        for (int j = 0; j < L; j++) {
            auto pointer = hash_map.find(make_pair((h_hashes1[i] + p1) % MOD1, (h_hashes2[i] + p2) % MOD2));
            if (pointer != hash_map.end()) {
                int vector_size = (int)pointer->second.size();
                for (int o = 0; o < vector_size; o++) {
                    printf("%d %d\n", i, pointer->second[o]);
                }
            }

            pointer = hash_map.find(make_pair((h_hashes1[i] - p1 + MOD1) % MOD1, (h_hashes2[i] - p2 + MOD2) % MOD2));
            if (pointer != hash_map.end()) {     
                int vector_size = (int)pointer->second.size();
                for (int o = 0; o < vector_size; o++) {
                    printf("%d %d\n", i, pointer->second[o]);
                }
            }

            p1 = (p1 * P1) % MOD1;
            p2 = (p2 * P2) % MOD2;
        }
        hash_map[make_pair(h_hashes1[i], h_hashes2[i])].push_back(i);
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) usage(argv[0]);
    int L, M;
    bool *h_input;
    long long int *h_hashes1,* h_hashes2;
    read_input(argv[1], L, M, h_input);


    h_hashes1 = new long long int[M];
    h_hashes2 = new long long int[M];
    if (h_hashes1 == NULL || h_hashes2 == NULL) ERR("operator new");
    memset(h_hashes1, 0, sizeof(long long int) * M);
    memset(h_hashes2, 0, sizeof(long long int) * M);

    calculate_hashes(h_hashes1, h_hashes2, h_input, L, M);
    find_hamming_one(h_hashes1, h_hashes2, L, M);

    return EXIT_SUCCESS;
}