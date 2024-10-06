#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

#define CORRECT_SOLUTION "\033[0;32mCORRECT SOLUTION!\033[1;37m"
#define BAD_SOLUTION "\033[0;31mBAD SOLUTION!\033[1;37m"

#define ERR(source) (fprintf(stderr,"%s:%d\n",__FILE__,__LINE__),\
                     perror(source),\
                     exit(EXIT_FAILURE))

void usage(char *name){
    fprintf(stderr,"USAGE: %s <correct_output_file_path> <solution_output_file_path>\n",name);
    exit(EXIT_FAILURE);
}

pair<int, int> create_pair(int a, int b) {
    if (a <= b) return make_pair(a, b);
    else return make_pair(b, a);
}

void read_input(char * file_path, vector<pair<int, int>>& data_vector) {
    int a, b;
    ifstream fileStream;    
    fileStream.open(file_path, ios::in);
    if (!fileStream.is_open()) ERR("ifstream.open");

    while(fileStream >> a) {
        if (fileStream.eof()) ERR("invalid output file");
        fileStream >> b;

        data_vector.push_back(create_pair(a, b));
    }
    sort(data_vector.begin(), data_vector.end());
}

void check_solution(vector<pair<int, int>>& correct_output, vector<pair<int, int>>& solution_output) {
    if ((int)correct_output.size() != (int)solution_output.size()) {
        cout << "\n" << BAD_SOLUTION << "\n\n";
        return;
    }
    int vectors_size = (int)correct_output.size();
    for (int i = 0; i < vectors_size; i++) {
        if (correct_output[i].first != solution_output[i].first || correct_output[i].second != solution_output[i].second) {
            cout << "\n" << BAD_SOLUTION << "\n\n";
            return;
        }
    }
    cout << "\n" << CORRECT_SOLUTION << "\n\n";
}

int main(int argc, char ** argv) {
    if (argc != 3) usage(argv[0]);
    
    vector<pair<int, int>> correct_output, solution_output;

    read_input(argv[1], correct_output);
    read_input(argv[2], solution_output);
    check_solution(correct_output, solution_output);
    return EXIT_SUCCESS;
}