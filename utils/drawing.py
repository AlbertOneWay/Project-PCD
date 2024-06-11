import matplotlib.pyplot as plt
import os
def draw_dotplot(dotplot, img_name='dotplot.jpg'):
    plt.figure(figsize=(10, 10))
    plt.imshow(dotplot, cmap='Greys', interpolation='none')
    plt.title(f'Dotplot de Secuencia 1 vs Secuencia 2')
    plt.xlabel('Secuencia 2')
    plt.ylabel('Secuencia 1')
    plt.savefig(img_name)
    #plt.show()

def plot_metrics(metrics, title_prefix, output_file_prefix):
    num_threads = [m[0] for m in metrics]
    times = [m[1] for m in metrics]
    speedups = [m[2] for m in metrics]
    efficiencies = [m[3] for m in metrics]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(num_threads, speedups, marker='o')
    plt.title(f'{title_prefix} - Aceleración')
    plt.xlabel('Número de Hilos/Procesos')
    plt.ylabel('Aceleración')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(num_threads, efficiencies, marker='o')
    plt.title(f'{title_prefix} - Eficiencia')
    plt.xlabel('Número de Hilos/Procesos')
    plt.ylabel('Eficiencia')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_file_prefix}_metrics.png')
    plt.show()


def graphics_times():
    metodos = ["secuencial", "multithreaded", "multiprocessing", "mpi", "pycuda"]
    tiempos_procesamiento = {metodo: [] for metodo in metodos}
    procesadores = [1, 2, 4, 8]

    # Leer el tiempo de procesamiento secuencial
    nombre_secuencial = f"results/sequential/results_time_sequential.txt"
    if os.path.exists(nombre_secuencial):
        with open(nombre_secuencial, "r") as file:
            for line in file:
                if "Tiempo de procesamiento" in line:
                    tiempo_secuencial = float(line.split(":")[1].strip().split()[0])
                    tiempos_procesamiento["secuencial"].append(tiempo_secuencial)
                    break

    # Leer tiempos de procesamiento para cada método
    for metodo in ["multithreaded", "multiprocessing", "mpi"]:
        for exp in range(0, 4):
            i = 2 ** exp
            nombre_archivo = f"results/{metodo}/results_time_{metodo}_{i}.txt"
            if os.path.exists(nombre_archivo):
                with open(nombre_archivo, "r") as file:
                    for line in file:
                        if "Tiempo de procesamiento" in line:
                            tiempo_procesamiento = float(line.split(":")[1].strip().split()[0])
                            tiempos_procesamiento[metodo].append(tiempo_procesamiento)
                            break
            else:
                print(f"El archivo '{nombre_archivo}' no existe.")
                tiempos_procesamiento[metodo].append(None)  # Añadir None para mantener el tamaño de la lista

    # Leer tiempo de procesamiento para PyCUDA
    nombre_pycuda = f"results/pycuda/results_time_pycuda.txt"
    if os.path.exists(nombre_pycuda):
        with open(nombre_pycuda, "r") as file:
            for line in file:
                if "Tiempo de procesamiento" in line:
                    tiempo_pycuda = float(line.split(":")[1].strip().split()[0])
                    tiempos_procesamiento["pycuda"].append(tiempo_pycuda)
                    break
    else:
        print(f"El archivo '{nombre_pycuda}' no existe.")
        tiempos_procesamiento["pycuda"].append(None)  # Añadir None si no existe el archivo
    
    # Graficar tiempos de procesamiento
    for metodo, tiempos in tiempos_procesamiento.items():
        if metodo == "secuencial" or metodo == "pycuda":
            plt.plot([1], tiempos, marker='o', label=f"Tiempo {metodo.capitalize()}")
        else:
            plt.plot(procesadores, tiempos, marker='o', label=f"Tiempo {metodo.capitalize()}")

    plt.xlabel('Número de Procesadores')
    plt.ylabel('Tiempo de Procesamiento (segundos)')
    plt.title('Comparación de Tiempos de Procesamiento')
    plt.grid(True)
    plt.legend()
    plt.show()