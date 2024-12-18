import os
import sys
from scipy.io import mmread
from scipy.sparse import bsr_matrix
import numpy as np

def save_bsr_matrix(matrix, filename, block_size=(4, 4)):
    """
    Save a bsr_matrix to a file in the specified custom format.

    Parameters:
        matrix (bsr_matrix): The BSR matrix to save.
        filename (str): Path to the file where the matrix will be saved.
    """
    if not isinstance(matrix, bsr_matrix):
        raise ValueError("Input matrix must be a bsr_matrix.")
    
    rows, cols = matrix.shape
    size = block_size[0]
    while rows % size != 0 and cols % size != 0:
        size = size // 2
    size= 1
    print(f"bsr using shape {size},{size}")
    try :
        matrix = matrix.tobsr((size, size), copy=True)
    except:
        try:
            matrix = matrix.tobsr((1, 1), copy=True)
        except:
            raise RuntimeError("wrong")

    # Extract BSR matrix data
    num_rows, num_cols = matrix.shape
    block_size = matrix.blocksize
    block_row_size, block_col_size = block_size
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr

    # Number of non-zero elements
    nnz = data.size

    # Number of blocks
    num_blocks = len(indices)

    with open(filename, 'w') as file:
        # Write the header information
        file.write(f"{num_rows} {num_cols} {nnz} {block_row_size} {block_col_size} {num_blocks}\n")

        # Write block offsets (indptr)
        file.write(" ".join(map(str, indptr)) + "\n")

        # Write block column indices
        file.write(" ".join(map(str, indices)) + "\n")

        # Write the values of the blocks
        for block in data:
            # print(block)
            file.write(" ".join(map(str, block.flatten())) + "\n")

        print(f"Saved BSR values to {filename}")

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
                    # matrix_csc = mmread(mtx_file_path).tocsc()
                    matrix_coo = mmread(mtx_file_path).tocoo()
                    matrix_ell = mmread(mtx_file_path).tocsr()
                    matrix_ell_col = mmread(mtx_file_path).tocsc()
                    matrix_bsr = matrix.tobsr()
                except Exception as e:
                    print(f"Error reading or converting {mtx_file_path}: {e}")
                    continue
                
                # Prepare the output file path
                base_name = os.path.splitext(file)[0]
                csr_file_path = os.path.join(root, f"{base_name}.csr")
                # csc_file_path = os.path.join(root, f"{base_name}.csc")
                coo_file_path = os.path.join(root, f"{base_name}.coo")
                bsr_file_path = os.path.join(root, f"{base_name}.bsr")
                ell_file_path_colind = os.path.join(root, f"{base_name}_colind.ell")
                ell_file_path_values = os.path.join(root, f"{base_name}_values.ell")
                ell_file_path_rowind = os.path.join(root, f"{base_name}_rowind.ell")
                ell_file_path_values_colmajor = os.path.join(root, f"{base_name}_values_colmajor.ell")
                
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
                    
                except Exception as e:
                    print(f"Error writing to {csr_file_path}: {e}")

                    '''
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
                    '''
                    
                try:
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
                    print(f"Error writing to {coo_file_path}: {e}")


                #########################
                ### ELLPACK ROW MAJOR ###
                #########################
                try:
                    ### Now write ELLpack format

                    rows, cols = matrix_ell.shape
                    nnz = matrix_ell.nnz

                    # now write the max num of non-zero in a row
                    max_nnz = matrix_ell.getnnz(axis=1).max()

                    colind = [[-1 for _ in range(max_nnz)] for _ in range(rows)]
                    values = [[0 for _ in range(max_nnz)] for _ in range(rows)]

                    for row in range(rows):
                        row_start = matrix_ell.indptr[row]
                        row_end = matrix_ell.indptr[row+1]

                        row_indices = matrix_ell.indices[row_start:row_end]
                        row_values = matrix_ell.data[row_start:row_end]

                        for i,col in enumerate(row_indices):
                            colind[row][i] = col
                            values[row][i] = row_values[i]

                    with open(ell_file_path_colind, "w") as colind_file:
                        colind_file.write(f"{rows} {cols} {nnz} {max_nnz}\n")

                        for row in colind:
                            colind_file.write(" ".join(map(str, row)) + "\n")

                        print(f"Saved ELL colind to {ell_file_path_colind}")

                except Exception as e:
                    print(f"Error writing to {ell_file-path_colin}: {e}")

                try:

                    with open(ell_file_path_values, "w") as value_file:
                        for row in values:
                            value_file.write(" ".join(map(str, row)) + "\n")
                        print(f"Saved ELL values to {ell_file_path_values}")
                except Exception as e:
                    print(f"Error writing to {ell_file_path_value}: {e}")
                    

                #########################
                ### ELLPACK COL MAJOR ###
                #########################
                try:
                    ### Now write ELLpack format

                    rows, cols = matrix_ell_col.shape
                    nnz = matrix_ell_col.nnz

                    # now write the max num of non-zero in a col
                    max_nnz = matrix_ell_col.getnnz(axis=1).max()

                    rowind = [[-1 for _ in range(max_nnz)] for _ in range(cols)]
                    values = [[0 for _ in range(max_nnz)] for _ in range(cols)]

                    for col in range(cols):
                        col_start = matrix_ell_col.indptr[col]
                        col_end = matrix_ell_col.indptr[col+1]

                        col_indices = matrix_ell_col.indices[col_start:col_end]
                        col_values = matrix_ell_col.data[col_start:col_end]

                        for i,row in enumerate(col_indices):
                            rowind[col][i] = row
                            values[col][i] = col_values[i]

                    with open(ell_file_path_rowind, "w") as rowind_file:
                        rowind_file.write(f"{rows} {cols} {nnz} {max_nnz}\n")

                        for col in rowind:
                            rowind_file.write(" ".join(map(str, col)) + "\n")

                        print(f"Saved ELL rowind to {ell_file_path_rowind}")

                except Exception as e:
                    print(f"Error writing to {ell_file-path_rowin}: {e}")

                try:

                    with open(ell_file_path_values_colmajor, "w") as value_file:
                        for col in values:
                            value_file.write(" ".join(map(str, col)) + "\n")
                        print(f"Saved ELL values to {ell_file_path_values_colmajor}")
                except Exception as e:
                    print(f"Error writing to {ell_file_path_values_colmajor}: {e}")


                try:
                    save_bsr_matrix(matrix_bsr, bsr_file_path)
                        
                except Exception as e:
                    print(f"Error writing to {csr_file_path}: {e}")

                print("finish all")


if __name__ == "__main__":
    # Check if a directory argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)
    
    # Get the directory from command line arguments
    input_directory = sys.argv[1]
    
    # Process the directory
    process_mtx(input_directory)