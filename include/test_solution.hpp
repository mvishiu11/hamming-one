#ifndef TEST_SOLUTION_HPP
#define TEST_SOLUTION_HPP

#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

#define CORRECT_SOLUTION "\033[0;32mCORRECT SOLUTION!\033[1;37m"
#define BAD_SOLUTION "\033[0;31mBAD SOLUTION!\033[1;37m"

void read_input(char * file_path, vector<pair<int, int>>& data_vector);

void check_solution(vector<pair<int, int>>& correct_output, vector<pair<int, int>>& solution_output);

#endif