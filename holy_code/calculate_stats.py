import numpy as np

###################################################################################################
    
    ## Calculating the minimum, maximum, and standard deviation for the matrices

#####################################################################################################

def calculate_statistics(data, matrix_name):
    if data.size > 0:
        min_val = np.min(data)
        max_val = np.max(data)
        std_dev = np.std(data)

        print(f"\nOverall Minimum of Matrix {matrix_name}: {min_val}")
        print(f"Overall Maximum of Matrix {matrix_name}: {max_val}")
        print(f"Overall Standard Deviation of Matrix {matrix_name}: {std_dev}")