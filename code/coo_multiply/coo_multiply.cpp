#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <filesystem>
#include <fstream>

bool endsWith(const std::string& fullString, const std::string& ending) {
    if (ending.size() > fullString.size())
        return false;

    // Compare the ending of the full string with the target
    // ending
    return fullString.compare(fullString.size()
                                  - ending.size(),
                              ending.size(), ending)
           == 0;
}

float** coo_multiply(std::ifstream& coo_stream, std::ifstream& dense_stream) {
    int rows_coo, cols_coo, nnz_coo;
    int rows_dense, cols_dense, nnz_dense;

    // Read COO matrix
    coo_stream >> rows_coo >> cols_coo >> nnz_coo;

    int *row_idx = new int[nnz_coo];
    int *col_idx = new int[nnz_coo];
    float *coo_values = new float[nnz_coo];

    for (int i = 0; i < nnz_coo; i++) {
        coo_stream >> row_idx[i] >> col_idx[i] >> coo_values[i];
    }


    // Read dense matrix
    dense_stream >> rows_dense >> cols_dense >> nnz_dense;

    if (cols_coo != rows_dense) {
        delete[] row_idx;
        delete[] col_idx;
        delete[] coo_values;
    }

    float **dense_matrix = new float*[rows_dense];

    for (int i = 0; i < rows_dense; i++) {
        dense_matrix[i] = new float[cols_dense];
        for (int j = 0; j < cols_dense; j++) {
            dense_stream >> dense_matrix[i][j];
        }
    }


    // Create result matrix
    float **result = new float*[rows_coo];
    for (int i = 0; i < rows_coo; i++) {
        result[i] = new float[cols_dense];
        std::memset(result[i], 0, cols_dense * sizeof(float));
    }


    // Multiply
    for (int i = 0; i < nnz_coo; i++) {
        int row = row_idx[i];
        int col = col_idx[i];
        float value = coo_values[i];

        for (int j = 0; j < cols_dense; j++) {
            result[row][j] += value * dense_matrix[col][j];
        }

    }


    // cleanup
    delete[] row_idx;
    delete[] col_idx;
    delete[] coo_values;
    for (int i = 0; i < rows_dense; i++) {
        delete[] dense_matrix[i];
    }
    delete[] dense_matrix;

    return result;
}

int main(int argc, char *argv[])
{
    std::string input_dirname;
    bool COO = true;
    bool CSR = false;


    int opt;
    while ((opt = getopt(argc, argv, "f:o:")) != -1) {
    switch (opt) {
        case 'f':
        input_dirname = optarg;
        break;

        case 'o':
        COO = true;
        break;

        case 'r':
        CSR = true;
        break;

        default:
        std::cerr << "Usage: " << argv[0] << " -f input_dirname [-o] [-r]\n";
        exit(EXIT_FAILURE);


    }
    }

    // check required argument
    if (empty(input_dirname)) {
        std::cerr << "Usage: " << argv[0] << " -f input_dirname [-o] [-r]\n";
        exit(EXIT_FAILURE);
    }

    std::string extension = ".coo";
    if (COO) {
        extension = ".coo";
    }
    else if (CSR) {
        extension = ".csr";
    }

    // find coo and dense files
    bool coo_found = false, dense_found = false;
    std::string coo_file, dense_file;

    for (const auto &entry: std::filesystem::directory_iterator(input_dirname)) {
        if (entry.is_regular_file()) {
            std::string file_name = entry.path().filename().string();

            if (endsWith( file_name, extension)) {
                coo_file = entry.path().string();
                coo_found = true;
                std::cout <<extension << " found: " << coo_file << "\n";
            }
            else if (endsWith(file_name, "dense.in")) {
                dense_file = entry.path().string();
                dense_found = true;
                std::cout << "dense found: " << dense_file << "\n";
            }
        }

    }

    if (!coo_found || !dense_found) {
        std::cerr << "Error: Missing required files *.coo and/or dense.in in directory: " << input_dirname << "\n";
        exit(EXIT_FAILURE);
    }

    // open files for processing
    std::ifstream coo_stream(coo_file);
    if (!coo_stream.is_open()) {
        std::cerr << "Error: Cannot open .coo file: "<< coo_file << "\n";
        exit(EXIT_FAILURE);

    }

    std::ifstream dense_stream(dense_file);
    if (!dense_stream.is_open()) {
        std::cerr << "Error: Cannot open dense.in file: "<< dense_file << "\n";
        exit(EXIT_FAILURE);

    }


    // Multiply
    std::cout << "Multiplying\n";
    float** result = coo_multiply(coo_stream, dense_stream);
    // size of result = rows_result x cols_result

    int rows_result, cols_coo;
    coo_stream.clear();
    coo_stream.seekg(0, std::ios_base::beg);
    coo_stream >> rows_result >> cols_coo;
    int rows_dense, cols_result;
    dense_stream.clear();
    dense_stream.seekg(0, std::ios_base::beg);
    dense_stream >> rows_dense >> cols_result;

    
    // Save result
    std::string out_file = input_dirname + "/result.out";
    std::cout<< "Saving result to " << out_file << "\n";

    std::ofstream result_file(out_file);

    if (!result_file.is_open()) {
        std::cerr << "Error: Could not save result to " << out_file << "\n";

        for (int row = 0; row < rows_result; row++) {
            delete[] result[row];
        }
        delete[] result;
        coo_stream.close();
        dense_stream.close();
        exit(EXIT_FAILURE);
    }

    //std::cout << "result:\n";
    for (int row = 0; row < rows_result; row++) {
        for (int col = 0; col < cols_result; col++) {
            //std::cout << result[row][col] << " ";
            result_file << result[row][col] << " ";
        }
        //std::cout << "\n";
        result_file << "\n";
        delete[] result[row];
    }


    // cleanup
    delete[] result;
    coo_stream.close();
    dense_stream.close();

    return 0;
}
