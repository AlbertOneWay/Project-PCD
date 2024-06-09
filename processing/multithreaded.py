import time
import numpy as np
import threading
from utils.file import read_FASTA
from utils.drawing import draw_dotplot
from utils.filter import apply_custom_filter

def worker(sequence1, sequence2, dotplot_matrix, start, end):
    """Función de trabajador para calcular una parte del dotplot."""
    for i in range(start, end):
        for j in range(len(sequence2)):
            if sequence1[i] == sequence2[j]:
                if i == j:
                    dotplot_matrix[i][j] = 1
                else:
                    dotplot_matrix[i][j] = 0.7
            else:
                dotplot_matrix[i][j] = 0
        print(f'Progreso: {i+1}/{end} filas completadas')

def generate_multithreaded_dotplot(file1, file2, num_threads=4):
    # Medir tiempo de carga de archivos
    start_load_time = time.time()
    seq1 = read_FASTA(file1)
    seq2 = read_FASTA(file2)

    sequence1 = seq1[0:20000]
    sequence2 = seq2[0:20000]
    end_load_time = time.time()

    load_time = end_load_time - start_load_time
    print(f"Tiempo de carga de archivos: {load_time} segundos")

    start_process_time = time.time()
    len_seq1 = len(sequence1)
    len_seq2 = len(sequence2)

    dotplot_matrix = np.zeros((len_seq1, len_seq2), dtype=np.float32)

    # Crear hilos
    threads = []
    chunk_size = len_seq1 // num_threads

    for i in range(num_threads):
        start = i * chunk_size
        end = len_seq1 if i == num_threads - 1 else (i + 1) * chunk_size
        thread = threading.Thread(target=worker, args=(sequence1, sequence2, dotplot_matrix, start, end))
        threads.append(thread)
        thread.start()

    # Esperar a que todos los hilos terminen
    for thread in threads:
        thread.join()

    end_process_time = time.time()
    process_time = end_process_time - start_process_time

    start_image_time = time.time()

    # Convertir la matriz a formato numpy para aplicar el filtro
    dotplot_matrix_np = np.array(dotplot_matrix, dtype=np.uint8)

    # Aplicar el filtro
    apply_custom_filter(dotplot_matrix_np, f"results/multithreaded/dotplot_filtered{num_threads}.jpg")

    draw_dotplot(dotplot_matrix, f'results/multithreaded/dotplot{num_threads}.jpg')

    end_image_time = time.time()
    image_time = end_image_time - start_image_time

    total_time = load_time + process_time + image_time
    return load_time, process_time, image_time, total_time
