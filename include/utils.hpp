#pragma once

#include <string>
#include <cstdint>
#include <iostream>

// global variable declared in main
extern std::string testcase;

#define REL_TOL 1e-2f
#define ABS_TOL 1e-3f    

inline bool endsWith(const std::string &fullString, const std::string &ending) {
    if (ending.size() > fullString.size())
        return false;

    // Compare the ending of the full string with the target
    // ending
    return fullString.compare(fullString.size() - ending.size(), ending.size(),
                              ending) == 0;
}


inline void reportTime(
    std::string testcase,
    uint32_t aNumRows, uint32_t aNumCols, uint32_t aNumNonZero, std::string format,
    int ordering,
    int kernelNum, double pro, double kernel, double epilog, bool correct) {
    std::string ord;
    if (ordering == 0) {
        ord = "ROW_MAJOR";
    } else {
        ord = "COL_MAJOR";
    }

    double total = pro + kernel + epilog;

    std::cout << "{\n\"testcase\":\"" << testcase << "\",\n" 
                << "\"sparsity\":\"" << ((double)aNumNonZero / (aNumRows * aNumCols)) << "\",\n"
                << "\"format\":\"" << format << "\",\n"
                << "\"kernelType\":\"" << kernelNum << "\",\n"
                << "\"denseOrdering\":\"" << ord << "\",\n"
                << "\"correct\":\"" << correct << "\",\n";
    printf("\"cudaPrologTimeMs\":\"%lf\",\n"
            "\"cudaKernelTimeMs\":\"%lf\",\n"
            "\"cudaEpilogTimeMs\":\"%lf\",\n"
            "\"cudaTotalTimeMs\":\"%lf\",\n"
            "\"sequentialTimeMs\":\"%lf\"\n},\n", pro, kernel, epilog, total, 0.f);
}