import os
import numpy as np

def gen_matrix(filepath, numRows, numCols):
    m = np.random.rand(numRows, numCols).astype(np.float32)
    np.savetxt(f"{filepath}/{numRows}x{numCols}.dns", m)
    
gen_matrix("code/data", 32, 32)