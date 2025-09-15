import numpy as np
import time

matrix = np.random.rand(1000, 1000)

start_row = time.time()
row_sums = np.sum(matrix, axis=1)
end_row = time.time()

start_column = time.time()
col_sums = np.sum(matrix, axis=0)
end_column = time.time()

print(f"Time to calculate the sum of each row: {end_row - start_row:.6f} s.")
print(f"Time to calculate the sum of each column: {end_column - start_column:.6f} s.")
