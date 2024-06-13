import time
import numpy as np
import multiprocessing as mp
from utils.file import read_FASTA
from utils.drawing import draw_dotplot
from utils.filter import apply_custom_filter
import gc

def worker_multiprocessing(sequence1_chunk, sequence2_numeric):
    """FunciÃ³n de trabajador para calcular una parte del dotplot."""
    dotplot_chunk = np.equal.outer(sequence1_chunk, sequence2_numeric).astype(np.uint8)
    return dotplot_chunk

def parallel_multiprocessing_dotplot(sequence1_numeric, sequence2_numeric, processes=mp.cpu_count()):
    len_seq1 = len(sequence1_numeric)
    chunk_size = len_seq1 // processes
    sequences = []

    for i in range(processes):
        start = i * chunk_size
        end = len_seq1 if i == processes - 1 else (i + 1) * chunk_size
        sequences.append(sequence1_numeric[start:end])

    with mp.Pool(processes=processes) as pool:
        dotplot_chunks = pool.starmap(worker_multiprocessing, [(sequences[i], sequence2_numeric) for i in range(processes)])
    
    dotplot_matrix = np.vstack(dotplot_chunks)
    return dotplot_matrix

def generate_multiprocessing_dotplot(file1, file2, num_processes=4):
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

    start_process_time = time.time()

    base_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    sequence1_numeric = np.array([base_mapping[base] for base in sequence1], dtype=np.uint8)
    sequence2_numeric = np.array([base_mapping[base] for base in sequence2], dtype=np.uint8)

    sequence1 = None
    sequence2 = None
    gc.collect()

    # Generar el dotplot en paralelo usando multiprocessing
    dotplot_matrix = parallel_multiprocessing_dotplot(sequence1_numeric, sequence2_numeric, num_processes)

    end_process_time = time.time()
    process_time = end_process_time - start_process_time

    start_image_time = time.time()

    # Aplicar el filtro
    apply_custom_filter(dotplot_matrix, f"results/multiprocessing/dotplot_filtered{num_processes}.jpg")
    draw_dotplot(dotplot_matrix, f'results/multiprocessing/dotplot{num_processes}.jpg')

    end_image_time = time.time()
    image_time = end_image_time - start_image_time

    total_time = load_time + process_time + image_time
    return load_time, process_time, image_time, total_time