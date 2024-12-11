#include "format.hpp"
#include "spmm_coo.hpp"
#include "spmm_csr.hpp"
#include "spmm_ell.hpp"
#include "utils.hpp"
#include <cstdlib>
#include <filesystem>
#include <string>

void printHelp(char *filename) {
    std::cout << "Usage: " << filename << " -f input_dirname [-c] [-o] [-r] [-e]\n";
    std::cout << "\t-f input_dirname: directory that contains source "
                 "matrices\n";
    std::cout << "\t-c: enable CUDA\n";
    std::cout << "\t-o: use coo input format\n";
    std::cout << "\t-r: use csr input format\n";
    std::cout << "\t-e: use ell input format\n";
    std::cout << "\tEither -o or -r must be supplied. If both "
                 "supplied, program use coo source format\n";
}

int main(int argc, char *argv[]) {
    std::string input_dirname;
    bool TEST_COO = false;
    bool TEST_CSR = false;
    bool TEST_ELL = false;
    bool CUDA = false;

    int opt;
    while ((opt = getopt(argc, argv, "f:orhce")) != -1) {
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
        case 'e':
            TEST_ELL = true;
            break;
        case 'c':
            CUDA = true;
            break;
        case 'h':
            printHelp(argv[0]);
            exit(0);
        default:
            std::cerr << "Usage: " << argv[0] << " -f input_dirname [-c] [-o] [-r] [-e]\n";
            exit(EXIT_FAILURE);
        }
    }

    // check required argument
    if (empty(input_dirname) || (!TEST_COO && !TEST_CSR && !TEST_ELL)) {
        printHelp(argv[0]);
        exit(EXIT_FAILURE);
    }

    // Find files
    bool coo_found = false, csr_found = false, dense_found = false;
    bool ell_colind_found = false, ell_values_found = false;

    std::string coo_file, csr_file, dense_file;
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

    cuspmm::DenseMatrix<float> *dense =
        new cuspmm::DenseMatrix<float>(dense_file);

    if (TEST_COO) {
        std::cout << "###COO,testCase:" << input_dirname << ',';
        cuspmm::SparseMatrixCOO<float> *a =
            new cuspmm::SparseMatrixCOO<float>(coo_file);

        float abs_tol = 1.0e-3f;
        double rel_tol = 1.0e-2f;

        cuspmm::runEngineCOO<float>(a, dense, abs_tol, rel_tol);
    }

    if (TEST_CSR) {
        std::cout << "###CSR,testCase:" << input_dirname << ',';
        cuspmm::SparseMatrixCSR<float> *a =
            new cuspmm::SparseMatrixCSR<float>(csr_file);

        float abs_tol = 1.0e-3f;
        double rel_tol = 1.0e-2f;

        cuspmm::runEngineCSR<float>(a, dense, abs_tol, rel_tol);
    }

    if (TEST_ELL) {
        std::cout << "###ELL,testCase:" << input_dirname << ',';
        cuspmm::SparseMatrixELL<float> *a =
            new cuspmm::SparseMatrixELL<float>(ell_colind_file, ell_values_file);

        float abs_tol = 1.0e-3f;
        double rel_tol = 1.0e-2f;

        cuspmm::runEngineELL<float>(a, dense, abs_tol, rel_tol);
    }

    return 0;
}
