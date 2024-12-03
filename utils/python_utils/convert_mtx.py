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
                    matrix_coo = mmread(mtx_file_path).tocoo()
                except Exception as e:
                    print(f"Error reading or converting {mtx_file_path}: {e}")
                    continue
                
                # Prepare the output file path
                base_name = os.path.splitext(file)[0]
                csr_file_path = os.path.join(root, f"{base_name}.csr")
                csc_file_path = os.path.join(root, f"{base_name}.csc")
                coo_file_path = os.path.join(root, f"{base_name}.coo")
                
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
                    
                    with open(coo_file_path, "w") as out_file:
                        # Line 1: Number of rows, columns, and non-zero elements
                        rows, cols = matrix_coo.shape
                        nnz = matrix_coo.nnz
                        out_file.write(f"{rows} {cols} {nnz}\n")
                        
                        rows, cols, values = matrix_coo.row, matrix_coo.col, matrix_coo.data

                        # Sort by row, then by column (row-major order)
                        sorted_indices = np.lexsort((cols, rows))  # Sort by rows, then by cols
                        rows = rows[sorted_indices]
                        cols = cols[sorted_indices]
                        values = values[sorted_indices]
                        
                        for r, c, v in zip(rows, cols, values):
                            out_file.write(f"{r} {c} {v}\n")
    
                    print(f"Saved COO format to {coo_file_path}")
                    
                        
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