import time
import numpy as np
import threading
from utils.file import read_FASTA
from utils.drawing import draw_dotplot
from utils.filter import apply_custom_filter
import gc

def worker(sequence1_chunk, sequence2_numeric, dotplot_matrix, start_index):
    """Función de trabajador para calcular una parte del dotplot."""
    dotplot_chunk = np.equal.outer(sequence1_chunk, sequence2_numeric).astype(np.uint8)
    dotplot_matrix[start_index:start_index + len(sequence1_chunk), :] = dotplot_chunk

def generate_multithreaded_dotplot(file1, file2, num_threads=4):
    # Medir tiempo de carga de archivos
    start_load_time = time.time()
    seq1 = read_FASTA(file1)
    seq2 = read_FASTA(file2)

    sequence1 = seq1[0:20000]
    sequence2 = seq2[0:20000]

    seq1 = None
    seq2 = None
    file1 = None
    file2 = None
    gc.collect()

    end_load_time = time.time()

    load_time = end_load_time - start_load_time
    print(f"Tiempo de carga de archivos: {load_time} segundos")

    # Convertir secuencias a arrays de números
    base_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    sequence1_numeric = np.array([base_mapping[base] for base in sequence1], dtype=np.uint8)
    sequence2_numeric = np.array([base_mapping[base] for base in sequence2], dtype=np.uint8)

    sequence1 = None
    sequence2 = None
    gc.collect()

    start_process_time = time.time()
    len_seq1 = len(sequence1_numeric)
    len_seq2 = len(sequence2_numeric)

    dotplot_matrix = np.zeros((len_seq1, len_seq2), dtype=np.uint8)

    # Crear hilos
    threads = []
    chunk_size = len_seq1 // num_threads

    for i in range(num_threads):
        start = i * chunk_size
        end = len_seq1 if i == num_threads - 1 else (i + 1) * chunk_size
        sequence1_chunk = sequence1_numeric[start:end]
        thread = threading.Thread(target=worker, args=(sequence1_chunk, sequence2_numeric, dotplot_matrix, start))
        threads.append(thread)
        thread.start()

    # Esperar a que todos los hilos terminen
    for thread in threads:
        thread.join()

    end_process_time = time.time()
    process_time = end_process_time - start_process_time

    start_image_time = time.time()

    # Aplicar el filtro
    apply_custom_filter(dotplot_matrix, f"results/multithreaded/dotplot_filtered{num_threads}.jpg")

    draw_dotplot(dotplot_matrix, f'results/multithreaded/dotplot{num_threads}.jpg')

    end_image_time = time.time()
    image_time = end_image_time - start_image_time

    total_time = load_time + process_time + image_time
    return load_time, process_time, image_time, total_time