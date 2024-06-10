import cv2
import numpy as np
import multiprocessing as mp

def process_chunk_with_margin(chunk, margin):
    # Agrega el margen a la chunk
    chunk_with_margin = np.pad(chunk, pad_width=((margin, margin), (margin, margin)), mode='constant', constant_values=0)
    
    diag_kernel = np.array([[1, -1, -1],
                            [-1, 1, -1],
                            [-1, -1, 1]])
    filtered_chunk_with_margin = cv2.filter2D(chunk_with_margin, -1, diag_kernel)
    
    # Elimina el margen del resultado
    filtered_chunk = filtered_chunk_with_margin[margin:-margin, margin:-margin]
    return filtered_chunk

def apply_custom_filter(matrix, output_path, num_processes=8):
    margin = 1  # Debe ser al menos el tamaño del kernel - 1

    # Divide la matriz en partes, con margen
    chunk_height = matrix.shape[0] // num_processes
    chunks = [matrix[i*chunk_height:(i+1)*chunk_height] for i in range(num_processes)]

    # Crea un pool de procesos
    with mp.Pool(processes=num_processes) as pool:
        # Procesa cada chunk en paralelo
        filtered_chunks = pool.starmap(process_chunk_with_margin, [(chunk, margin) for chunk in chunks])

    # Combina los resultados
    filtered_matrix = np.vstack(filtered_chunks)

    # Normaliza la matriz combinada
    normalized_matrix = cv2.normalize(filtered_matrix, None, 0, 127, cv2.NORM_MINMAX)

    # Aplica el umbral
    threshold_level = 50
    _, binary_matrix = cv2.threshold(normalized_matrix, threshold_level, 255, cv2.THRESH_BINARY)

    # Guarda y muestra la matriz binaria
    cv2.imwrite(output_path, binary_matrix)
    cv2.imshow('Filtered Dotplot', binary_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()