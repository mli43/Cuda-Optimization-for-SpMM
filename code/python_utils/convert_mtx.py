import os
import sys
from scipy.io import mmread
import numpy as np


def process_mtx(directory):
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "dense.mtx":
                dense_mtx_path = os.path.join(root, file)
                
                # Read the dense.mtx file
                try:
                    matrix = mmread(dense_mtx_path)
                    dense_matrix = matrix.todense()
                except Exception as e:
                    print(f"Error reading {dense_mtx_path}: {e}")
                    continue
                
                # Prepare the output file path
                output_file_path = os.path.join(root, "dense.in")
                
                # Write the matrix to dense.in
                try:
                    with open(output_file_path, "w") as out_file:
                        rows, cols = dense_matrix.shape
                        non_zero_elements = np.count_nonzero(dense_matrix)
                        
                        # Write the matrix dimensions and number of non-zero elements
                        out_file.write(f"{rows} {cols} {non_zero_elements}\n")
                        
                        # Write each row
                        for row in dense_matrix:
                            out_file.write(" ".join(map(str, row.A1)) + "\n")
                    
                    print(f"Processed {dense_mtx_path} -> {output_file_path}")
                except Exception as e:
                    print(f"Error writing {output_file_path}: {e}")
                    
            elif file.endswith(".mtx"):
                mtx_file_path = os.path.join(root, file)
                
                # Convert the matrix to CSR and CSC format
                try:
                    print(f"Processing {mtx_file_path}...")
                    matrix = mmread(mtx_file_path).tocsr()
                    matrix_csc = mmread(mtx_file_path).tocsc()
                except Exception as e:
                    print(f"Error reading or converting {mtx_file_path}: {e}")
                    continue
                
                # Prepare the output file path
                base_name = os.path.splitext(file)[0]
                csr_file_path = os.path.join(root, f"{base_name}.csr")
                csc_file_path = os.path.join(root, f"{base_name}.csc")
                
                # Save the CSR matrix in the specified format
                try:
                    with open(csr_file_path, "w") as out_file:
                        # Line 1: Number of rows, columns, and non-zero elements
                        rows, cols = matrix.shape
                        nnz = matrix.nnz
                        out_file.write(f"{rows} {cols} {nnz}\n")
                        
                        # Line 2: row_ptr array
                        row_ptr = matrix.indptr
                        out_file.write(" ".join(map(str, row_ptr)) + "\n")
                        
                        # Line 3: col_idx array
                        col_idx = matrix.indices
                        out_file.write(" ".join(map(str, col_idx)) + "\n")
                        
                        # Line 4: values array
                        values = matrix.data
                        out_file.write(" ".join(map(str, values)) + "\n")
                    
                    print(f"Saved CSR format to {csr_file_path}")
                    
                    with open(csc_file_path, "w") as out_file:
                        # Line 1: Number of rows, columns, and non-zero elements
                        rows, cols = matrix_csc.shape
                        nnz = matrix_csc.nnz
                        out_file.write(f"{rows} {cols} {nnz}\n")
                        
                        # Line 2: col_ptr array
                        col_ptr = matrix_csc.indptr
                        out_file.write(" ".join(map(str, col_ptr)) + "\n")
                        
                        # Line 3: row_idx array
                        row_idx = matrix_csc.indices
                        out_file.write(" ".join(map(str, row_idx)) + "\n")
                        
                        # Line 4: values array
                        values = matrix_csc.data
                        out_file.write(" ".join(map(str, values)) + "\n")
                    
                    print(f"Saved CSC format to {csc_file_path}")
                except Exception as e:
                    print(f"Error writing to {csr_file_path}: {e}")


if __name__ == "__main__":
    # Check if a directory argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)
    
    # Get the directory from command line arguments
    input_directory = sys.argv[1]
    
    # Process the directory
    process_mtx(input_directory)