import time
import numpy as np
from utils.file import read_FASTA
from utils.drawing import draw_dotplot
from utils.filter import apply_custom_filter

def generate_sequential_dotplot(file1, file2):

    #Medir tiempo de carga de archivos
    start_time = time.time()
    seq1 = read_FASTA(file1)
    seq2 = read_FASTA(file2)

    sequence1 = seq1[0:20000]
    sequence2 = seq2[0:20000]
    end_time = time.time()

    load_time = end_time - start_time

     # Convertir secuencias a arrays de números
    base_mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    sequence1_numeric = np.array([base_mapping[base] for base in sequence1], dtype=np.uint8)
    sequence2_numeric = np.array([base_mapping[base] for base in sequence2], dtype=np.uint8)

    print(f"Tiempo de carga de archivos: {load_time} segundos")

    start_process = time.time()

    dotplot_matrix = np.equal.outer(sequence1_numeric, sequence2_numeric).astype(np.uint8)
            
        ##print(f'Progreso: {i+1}/{len_seq1} filas completadas')
    end_process = time.time()
    process_time = end_process - start_process

    # Aplicar el filtro
    apply_custom_filter(dotplot_matrix, "results/sequential/dotplot_filtered.jpg")
    draw_dotplot(dotplot_matrix, 'results/sequential/dotplot.jpg')

    return load_time, process_time
   