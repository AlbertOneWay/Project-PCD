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

    sequence1 = seq1[0:10000]
    sequence2 = seq2[0:10000]
    end_time = time.time()

    load_time = end_time - start_time
    print(f"Tiempo de carga de archivos: {load_time} segundos")

    start_process = time.time()
    len_seq1 = len(sequence1)
    len_seq2 = len(sequence2)

    dotplot_matrix = np.zeros((len_seq1, len_seq2), dtype=np.float32)

    for i in range(len_seq1):
        for j in range(len_seq2):
            if sequence1[i] == sequence2[j]:
                if i == j:
                    dotplot_matrix[i][j] = 1
                else:
                    dotplot_matrix[i][j]  = 0.7
            else:
                dotplot_matrix[i][j]  = 0
            
        print(f'Progreso: {i+1}/{len_seq1} filas completadas')
    end_process = time.time()
    process_time = end_process - start_process
    
     # Convertir la matriz a formato numpy para aplicar el filtro
    dotplot_matrix_np = np.array(dotplot_matrix, dtype=np.uint8)

    # Aplicar el filtro
    apply_custom_filter(dotplot_matrix_np, "results/sequential/dotplot_filtered.jpg")
    draw_dotplot(dotplot_matrix, 'results/sequential/dotplot.jpg')

    return load_time, process_time
   