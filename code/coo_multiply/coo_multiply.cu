#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 256

/*
 * Helper: checks if fullstring ends with ending. Used to check file extension
 */
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

__global__ void coo_multiply_kernel(
    const int* row_idx,
    const int* col_idx,
    const float* coo_values,
    const float* dense_matrix,
    float* result,
    int nnz_coo,
    int cols_dense)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes one non-zero element in the COO matrix
    if (idx < nnz_coo) {
        int row = row_idx[idx];
        int col = col_idx[idx];
        float value = coo_values[idx];

        for (int j = 0; j < cols_dense; j++) {
            atomicAdd(&result[row * cols_dense + j], value * dense_matrix[col * cols_dense + j]);
        }
    }
}

void coo_multiply_cuda(
    const int* h_row_idx,
    const int* h_col_idx,
    const float* h_coo_values,
    float** h_dense_matrix,
    float** h_result,
    int nnz_coo,
    int rows_coo,
    int cols_dense,
    int rows_dense)
{
    // Flatten dense_matrix and result arrays
    float* h_dense_flattened = new float[rows_dense * cols_dense];
    float* h_result_flattened = new float[rows_coo * cols_dense]();

    for (int i = 0; i < rows_dense; i++) {
        for (int j = 0; j < cols_dense; j++) {
            h_dense_flattened[i * cols_dense + j] = h_dense_matrix[i][j];
        }
    }

    // Device memory pointers
    int *d_row_idx, *d_col_idx;
    float *d_coo_values, *d_dense_flattened, *d_result_flattened;

    auto start = std::chrono::high_resolution_clock::now();

    // Allocate device memory
    cudaMalloc(&d_row_idx, nnz_coo * sizeof(int));
    cudaMalloc(&d_col_idx, nnz_coo * sizeof(int));
    cudaMalloc(&d_coo_values, nnz_coo * sizeof(float));
    cudaMalloc(&d_dense_flattened, rows_dense * cols_dense * sizeof(float));
    cudaMalloc(&d_result_flattened, rows_coo * cols_dense * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_row_idx, h_row_idx, nnz_coo * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, nnz_coo * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coo_values, h_coo_values, nnz_coo * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dense_flattened, h_dense_flattened, rows_dense * cols_dense * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_result_flattened, 0, rows_coo * cols_dense * sizeof(float));

    // Launch kernel
    auto start2 = std::chrono::high_resolution_clock::now();

    const int blocksPerGrid = (nnz_coo + BLOCKSIZE - 1) / BLOCKSIZE;
    coo_multiply_kernel<<<blocksPerGrid, BLOCKSIZE>>>(
        d_row_idx, d_col_idx, d_coo_values, d_dense_flattened, d_result_flattened, nnz_coo, cols_dense);
    cudaDeviceSynchronize();

    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);

    // Copy result back to host
    cudaMemcpy(h_result_flattened, d_result_flattened, rows_coo * cols_dense * sizeof(float), cudaMemcpyDeviceToHost);

    // Display time
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "CUDA time (ms): " << duration.count() << "\n";
    std::cout << "\tKernel time (ns): " << duration2.count() << "\n";

    // Reshape flattened result into 2D array
    for (int i = 0; i < rows_coo; i++) {
        for (int j = 0; j < cols_dense; j++) {
            h_result[i][j] = h_result_flattened[i * cols_dense + j];
        }
    }

    // Free device memory
    cudaFree(d_row_idx);
    cudaFree(d_col_idx);
    cudaFree(d_coo_values);
    cudaFree(d_dense_flattened);
    cudaFree(d_result_flattened);

    // Free host temporary memory
    delete[] h_dense_flattened;
    delete[] h_result_flattened;
}


/* 
 * This function takes opened coo and dense matrix and multiplies the matrices
 * sequentially
 */
float** coo_multiply(
std::ifstream& coo_stream, 
std::ifstream& dense_stream,
bool CUDA)
{
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
        std::cerr << "Error: Matrix dimension sparse_col != dense_row\n";
        return NULL;
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
    if (CUDA) {
        // parallely compute multiplication output
        coo_multiply_cuda(
            row_idx,
            col_idx,
            coo_values,
            dense_matrix,
            result,
            nnz_coo,
            rows_coo,
            cols_dense,
            rows_dense
        );
    }
    else {
        // sequentially compute multiplicaiton output
        float dummy = 0.0f;  // variable to prevent compiler from optimization and mess up timing
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < nnz_coo; i++) {
            int row = row_idx[i];
            int col = col_idx[i];
            float value = coo_values[i];

            for (int j = 0; j < cols_dense; j++) {
                result[row][j] += value * dense_matrix[col][j];
            }

        }
        dummy += result[0][0];

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        // use result to prevent compiler from 
        std::cout << "Sequential time (ns): " << duration.count() << "\n";
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

/* 
 * This is the main entry point of the program. It checks required argument -f
 * that need to specify directory name, checks the -o or -r flags to determine
 * which format to use, finds the input files and dense file, opens the files,
 * multiplies the matrix, and saves the result
 */
int main(int argc, char *argv[])
{
    std::string input_dirname;
    bool COO = false;
    bool CSR = false;
    bool CUDA = false;


    int opt;
    while ((opt = getopt(argc, argv, "f:orhc")) != -1) {
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

        case 'c':
        CUDA = true;
        break;

        case 'h':
        std::cout << "Usage: " << argv[0] << " -f input_dirname [-o] [-r]\n";
        std::cout << "\t-f input_dirname: directory that contains source matrices\n";
        std::cout << "\t-o: use coo input format\n";
        std::cout << "\t-r: use csr input format [NOT IMPLEMENTED]\n";
        std::cout << "\tEither -o or -r must be supplied. If both supplied, program use coo source format\n";
        exit(0);

        default:
        std::cerr << "Usage: " << argv[0] << " -f input_dirname [-o] [-r]\n";
        exit(EXIT_FAILURE);


    }
    }

    // check required argument
    if (empty(input_dirname) || (!COO && !CSR)) {
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
    float** result = coo_multiply(coo_stream, dense_stream, CUDA);
    // size of result = rows_result x cols_result
    if (result == NULL) {
        coo_stream.close();
        dense_stream.close();
        exit(EXIT_FAILURE);
    }

    int rows_result, cols_coo;
    coo_stream.clear();
    coo_stream.seekg(0, std::ios_base::beg);
    coo_stream >> rows_result >> cols_coo;
    int rows_dense, cols_result;
    dense_stream.clear();
    dense_stream.seekg(0, std::ios_base::beg);
    dense_stream >> rows_dense >> cols_result;

    
    // Save result
    std::string out_file; // = input_dirname + "/result.out";
    if (COO) {
        out_file = input_dirname + "/coo";
    }
    else if (CSR) {
        out_file = input_dirname + "/csr";
    }

    if (CUDA) {
        out_file = out_file + "_cuda.out";
    }
    else {
        out_file = out_file + ".out";
    }

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
