#ifndef TEST_SOLUTION_HPP
#define TEST_SOLUTION_HPP

#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

#define CORRECT_SOLUTION "\033[0;32mCORRECT SOLUTION!\033[1;37m"
#define BAD_SOLUTION "\033[0;31mBAD SOLUTION!\033[1;37m"

/**
 * @brief Reads pairs of integers from a file into a vector.
 * 
 * This function reads pairs of integers from the specified file and stores them
 * in the provided `data_vector`. Each pair is read as two integers on the same
 * line. The pairs are sorted after being read.
 * 
 * @param file_path Path to the file containing the pairs of integers.
 * @param data_vector Reference to the vector where the pairs will be stored.
 */
void read_input(char *file_path, vector<pair<int, int>>& data_vector);

/**
 * @brief Checks if the solution output matches the correct output.
 * 
 * This function compares the correct output with the provided solution output.
 * If the two vectors of pairs differ in size or content, it prints a "BAD SOLUTION"
 * message. Otherwise, it prints a "CORRECT SOLUTION" message.
 * 
 * @param correct_output Vector containing the correct pairs of integers.
 * @param solution_output Vector containing the solution's pairs of integers.
 */
void check_solution(vector<pair<int, int>>& correct_output, vector<pair<int, int>>& solution_output);

#endif // TEST_SOLUTION_HPP