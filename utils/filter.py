from scipy.signal import convolve2d
import numpy as np
import multiprocessing as mp
from utils.drawing import draw_dotplot

def find_diagonals(matrix):
    kernel = np.eye(3, dtype=int)  # Kernel para encontrar diagonales
    convolved = convolve2d(matrix, kernel, mode='valid')
    filtered_matrix = (convolved == 3).astype(int)
    return filtered_matrix

def process_chunk_with_margin(chunk, margin):
    chunk_with_margin = np.pad(chunk, pad_width=((margin, margin), (margin, margin)), mode='constant', constant_values=0)
    filtered_chunk = find_diagonals(chunk_with_margin)
    return filtered_chunk

def apply_custom_filter(matrix, outpath, num_processes=8):
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    margin = 1  

    chunk_height = matrix.shape[0] // num_processes
    chunks = [matrix[i*chunk_height:(i+1)*chunk_height] for i in range(num_processes)]

    with mp.Pool(processes=num_processes) as pool:
        filtered_chunks = pool.starmap(process_chunk_with_margin, [(chunk, margin) for chunk in chunks])

    filtered_matrix = np.vstack(filtered_chunks)
    
    draw_dotplot(filtered_matrix, outpath)