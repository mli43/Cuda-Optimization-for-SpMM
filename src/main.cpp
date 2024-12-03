#include "format.hpp"
#include "spmm_coo.hpp"
#include "spmm_csr.hpp"
#include "utils.hpp"
#include <cstdlib>
#include <filesystem>
#include <string>

void printHelp(char *filename) {
    std::cout << "Usage: " << filename << " -f input_dirname [-o] [-r]\n";
    std::cout << "\t-f input_dirname: directory that contains source "
                 "matrices\n";
    std::cout << "\t-o: use coo input format\n";
    std::cout << "\t-r: use csr input format [NOT IMPLEMENTED]\n";
    std::cout << "\tEither -o or -r must be supplied. If both "
                 "supplied, program use coo source format\n";
}

int main(int argc, char *argv[]) {
    std::string input_dirname;
    bool TEST_COO = false;
    bool TEST_CSR = false;
    bool CUDA = false;

    int opt;
    while ((opt = getopt(argc, argv, "f:orhc")) != -1) {
        switch (opt) {
        case 'f':
            input_dirname = optarg;
            break;
        case 'o':
            TEST_COO = true;
            break;
        case 'r':
            TEST_CSR = true;
            break;
        case 'c':
            CUDA = true;
            break;
        case 'h':
            printHelp(argv[0]);
            exit(0);
        default:
            std::cerr << "Usage: " << argv[0]
                      << " -f input_dirname [-o] [-r]\n";
            exit(EXIT_FAILURE);
        }
    }

    // check required argument
    if (empty(input_dirname) || (!TEST_COO && !TEST_CSR)) {
        printHelp(argv[0]);
        exit(EXIT_FAILURE);
    }

    // Find files
    bool coo_found = false, csr_found = false, dense_found = false;
    std::string coo_file, csr_file, dense_file;
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
            } else if (TEST_CSR && endsWith(file_name, "dense.in")) {
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
    if (!dense_found) {
        std::cerr << "Error: Missing required file dense.in in "
                  << input_dirname << "\n";
        exit(EXIT_FAILURE);
    }

    cuspmm::DenseMatrix<float> *dense =
        new cuspmm::DenseMatrix<float>(dense_file);

    if (TEST_COO) {

        cuspmm::SparseMatrixCOO<float> *a =
            new cuspmm::SparseMatrixCOO<float>(coo_file);

        float abs_tol = 1.0e-3f;
        double rel_tol = 1.0e-2f;

        cuspmm::runEngineCOO<float>(a, dense, abs_tol, rel_tol);
    }

    if (TEST_CSR) {

        cuspmm::SparseMatrixCSR<float> *a =
            new cuspmm::SparseMatrixCSR<float>(csr_file);

        float abs_tol = 1.0e-3f;
        double rel_tol = 1.0e-2f;

        cuspmm::runEngineCSR<float>(a, dense, abs_tol, rel_tol);
    }

    return 0;
}