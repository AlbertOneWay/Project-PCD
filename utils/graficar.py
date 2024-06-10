import os
import matplotlib.pyplot as plt

def graphics():
    tiempos_procesamiento = []
    procesadores = [1, 2, 4, 8]

    nombre_secuencial = f"results/sequential/results_time_sequential.txt"

    if os.path.exists(nombre_secuencial):
        with open(nombre_secuencial, "r") as file:
                for line in file:
                    if "Tiempo de procesamiento" in line:
                        tiempo_secuencial = float(line.split(":")[1].strip().split()[0])
                        break

    for exp in range(0, 4):  
        i = 2 ** exp
        nombre_archivo = f"results/mpi/results_time_mpi_{i}.txt"
        if os.path.exists(nombre_archivo):
            with open(nombre_archivo, "r") as file:
                for line in file:
                    if "Tiempo de procesamiento" in line:
                        tiempo_procesamiento = float(line.split(":")[1].strip().split()[0])
                        tiempos_procesamiento.append(tiempo_procesamiento)
                        break
        else:
            print(f"El archivo '{nombre_archivo}' no existe.")

    aceleraciones = []
    for i in tiempos_procesamiento:
        aceleraciones.append(tiempo_secuencial/i)

    plt.plot(procesadores, tiempos_procesamiento, marker='o', label="Speed-up MPI4PY")

    plt.xlabel('Número de Procesadores')
    plt.ylabel('Tiempo de Procesamiento (segundos)')
    plt.title('Tiempo de Procesamiento usando MPI4PY')

    plt.grid(True)
    plt.show()

    plt.plot(procesadores, aceleraciones, marker='o', color='r', label="Aceleracion MPI4PY")

    plt.xlabel('Número de Procesadores')
    plt.ylabel('Aceleración')
    plt.title('Aceleración en MPI4PY')

    plt.grid(True)
    plt.show()

