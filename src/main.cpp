#include "commons.hpp"
#include "engine/engine_coo.hpp"
#include "format.hpp"
#include "utils.hpp"
#include "engine.hpp"
#include "getopt.h"
#include <algorithm>
#include <bits/getopt_core.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>

void printHelp(char *filename) {
    std::cout << "Usage: " << filename << " [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  --bsr           Process data in Block Sparse Row format\n";
    std::cout << "  --coo           Process data in Coordinate format\n";
    std::cout << "  --csr           Process data in Compressed Sparse Row format\n";
    std::cout << "  --ell           Process data in ELLPACK format\n";
    std::cout << "  --cuda          Enable CUDA processing\n";
    std::cout << "  -d <directory>  Data directory\n";
    std::cout << "  -h, --help      Display this help message\n";
}

int main(int argc, char *argv[]) {
    std::string input_dirname;
    bool TEST_COO = false, TEST_CSR = false, TEST_BSR = false, TEST_ELL = false;
    bool CUDA = false;

    const struct option long_options[] = {
        {"bsr", no_argument, nullptr, 0},
        {"coo", no_argument, nullptr, 0},
        {"csr", no_argument, nullptr, 0},
        {"ell", no_argument, nullptr, 0},
        {"cuda", no_argument, nullptr, 0},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "hd:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 0:
                if (std::string(long_options[option_index].name) == "bsr") {
                    std::cout << "Option --bsr selected.\n";
                    TEST_BSR = true;
                } else if (std::string(long_options[option_index].name) == "coo") {
                    std::cout << "Option --coo selected.\n";
                    TEST_COO = true;
                } else if (std::string(long_options[option_index].name) == "csr") {
                    std::cout << "Option --csr selected.\n";
                    TEST_CSR = true;
                } else if (std::string(long_options[option_index].name) == "ell") {
                    std::cout << "Option --ell selected.\n";
                    TEST_ELL = true;
                } else if (std::string(long_options[option_index].name) == "cuda") {
                    std::cout << "Option --cuda selected.\n";
                    CUDA = true;
                }
                break;
            case 'h':
                printHelp(argv[0]);
                return 0;
            case 'd':
                input_dirname = optarg;
                std::cout << "Data directory set as " << input_dirname << "\n";
                break;
            case '?':
                // getopt_long already prints an error message.
                return 1;
            default:
                break;
        }
    }

    // check required argument
    if (empty(input_dirname) || (!TEST_COO && !TEST_CSR && !TEST_BSR && !TEST_ELL)) {
        printHelp(argv[0]);
        exit(EXIT_FAILURE);
    }

    // Find files
    bool coo_found = false, csr_found = false, bsr_found = false, dense_found = false;
    bool ell_colind_found = false, ell_values_found = false;
    std::string coo_file, csr_file, bsr_file, dense_file;
    std::string ell_colind_file, ell_values_file;

    for (const auto &entry :
         std::filesystem::directory_iterator(input_dirname)) {
        if (entry.is_regular_file()) {
            std::string file_name = entry.path().filename().string();

            if (TEST_COO && endsWith(file_name, ".coo")) {
                coo_file = entry.path().string();
                coo_found = true;
                std::cout << ".coo file is found: " << coo_file << "\n";
            } else if (TEST_CSR && endsWith(file_name, ".csr")) {
                csr_file = entry.path().string();
                csr_found = true;
                std::cout << ".csr file is found: " << csr_file << "\n";
            } else if (TEST_BSR && endsWith(file_name, ".bsr")) {
                bsr_file = entry.path().string();
                bsr_found = true;
                std::cout << ".bsr file is found: " << bsr_file << "\n";
            } else if (TEST_ELL && endsWith(file_name, "_colind.ell")) {
                ell_colind_file = entry.path().string();
                ell_colind_found = true;
                std::cout << "ell column index file is found: " << ell_colind_file << "\n";
            } else if (TEST_ELL && endsWith(file_name, "_values.ell")) {
                ell_values_file = entry.path().string();
                ell_values_found = true;
                std::cout << "ell values file is found: " << ell_values_file << "\n";
            } else if (endsWith(file_name, "dense.in")) {
                dense_file = entry.path().string();
                dense_found = true;
                std::cout << "dense file is found: " << dense_file << "\n";
            }
        }
    }

    if (TEST_COO && !coo_found) {
        std::cerr << "Error: Missing required files *.coo in " << input_dirname
                  << "\n";
        exit(EXIT_FAILURE);
    }
    if (TEST_CSR && !csr_found) {
        std::cerr << "Error: Missing required files *.csr in " << input_dirname
                  << "\n";
        exit(EXIT_FAILURE);
    }
    if (TEST_BSR && !bsr_found) {
        std::cerr << "Error: Missing required files *.bsr in " << input_dirname
                  << "\n";
    }
    if (TEST_ELL && (!ell_colind_found || !ell_values_found)) {
        std::cerr << "Error: Missing required files *_colind.ell and/or *_values.ell in " 
                  << input_dirname << "\n";
        exit(EXIT_FAILURE);
    }
    if (!dense_found) {
        std::cerr << "Error: Missing required file dense.in in "
                  << input_dirname << "\n";
        exit(EXIT_FAILURE);
    }

    float abs_tol = 1.0e-3f;
    double rel_tol = 1.0e-2f;
    auto *dense = new cuspmm::DenseMatrix<float, uint32_t>(dense_file);

    if (TEST_COO) {
        std::cout << "###COO,testCase:" << input_dirname << '\n';
        auto* a = new cuspmm::SparseMatrixCOO<float, uint32_t>(coo_file);
        auto* engine = new cuspmm::EngineCOO<float, uint32_t, double>(input_dirname);

        cuspmm::runEngine(engine, a, dense, abs_tol, rel_tol);
    }

    if (TEST_CSR) {
        std::cout << "###CSR,testCase:" << input_dirname << '\n';
        auto* a = new cuspmm::SparseMatrixCSR<float, uint32_t>(csr_file);
        auto* engine = new cuspmm::EngineCSR<float, uint32_t, double>(input_dirname);

        cuspmm::runEngine(engine, a, dense, abs_tol, rel_tol);
    }

    if (TEST_BSR) {
        std::cout << "###BSR,testCase:" << input_dirname << '\n';
        auto* a = new cuspmm::SparseMatrixBSR<float, uint32_t>(bsr_file);
        auto* engine = new cuspmm::EngineBSR<float, uint32_t, double>(input_dirname);

        cuspmm::runEngine(engine, a, dense, abs_tol, rel_tol);

    }

    if (TEST_ELL) {
        std::cout << "###ELL,testCase:" << input_dirname << '\n';
        auto* a = new cuspmm::SparseMatrixELL<float, uint32_t>(ell_colind_file, ell_values_file);
        auto* engine = new cuspmm::EngineELL<float, uint32_t, double>(input_dirname);

        cuspmm::runEngine(engine, a, dense, abs_tol, rel_tol);
    }

    return 0;
}
