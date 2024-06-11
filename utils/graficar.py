import os
import matplotlib.pyplot as plt
##import pycuda.driver as cuda

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
    eficiencias = []
    for idx, tiempo_procesamiento in enumerate(tiempos_procesamiento):
        aceleracion = tiempo_secuencial / tiempo_procesamiento
        aceleraciones.append(aceleracion)
        eficiencia = aceleracion / procesadores[idx]
        eficiencias.append(eficiencia)

    plt.plot(procesadores, tiempos_procesamiento, marker='o', label="Speed-up MPI4PY")
    plt.xlabel('Número de Procesadores')
    plt.ylabel('Tiempo de Procesamiento (segundos)')
    plt.title('Tiempo de Procesamiento usando MPI4PY')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/mpi_metrics_speedup.jpg')
    plt.show()

    plt.plot(procesadores, aceleraciones, marker='o', color='r', label="Aceleracion MPI4PY")
    plt.xlabel('Número de Procesadores')
    plt.ylabel('Aceleración')
    plt.title('Aceleración en MPI4PY')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/mpi_metrics_aceleracion.jpg')
    plt.show()

    plt.plot(procesadores, eficiencias, marker='o', color='g', label="Eficiencia MPI4PY")
    plt.xlabel('Número de Procesadores')
    plt.ylabel('Eficiencia')
    plt.title('Eficiencia en MPI4PY')
    plt.grid(True)
    plt.legend()
    plt.savefig('results/mpi_metrics_eficiencia.jpg')
    plt.show()
"""
def get_cuda_cores():
    cuda.init()
    device = cuda.Device(0)
    return device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT) * \
           device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR)

def graph_pycuda():
    tiempos_procesamiento = []
    nombre_secuencial = f"results/sequential/results_time_sequential.txt"
    tiempo_secuencial = 0

    if os.path.exists(nombre_secuencial):
        with open(nombre_secuencial, "r") as file:
            for line in file:
                if "Tiempo de procesamiento" in line:
                    tiempo_secuencial = float(line.split(":")[1].strip().split()[0])
                    tiempos_procesamiento.append(tiempo_secuencial)
                    break
    else:
        print(f"El archivo '{nombre_secuencial}' no existe.")
    
    nombre_archivo = f"results/pycuda/results_time_pycuda.txt"
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
    eficiencias = []
    cuda_cores = get_cuda_cores()

    for idx, tiempo_procesamiento in enumerate(tiempos_procesamiento):
        aceleracion = tiempo_secuencial / tiempo_procesamiento
        aceleraciones.append(aceleracion)
        if idx == 0:
            eficiencia = aceleracion / 1
        else:
            eficiencia = aceleracion / cuda_cores  # Usamos núcleos CUDA para PyCUDA
        eficiencias.append(eficiencia)

    labels = ['Secuencial', 'PyCUDA']
    plt.bar(labels, tiempos_procesamiento, color=['green', 'orange'])
    plt.title('Tiempo de Procesamiento usando PyCUDA')
    plt.grid(True)
    for i, v in enumerate(tiempos_procesamiento):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')
    plt.savefig('Tiempo.jpg')
    plt.clf()

    labels = ['Secuencial', 'PyCUDA']
    plt.bar(labels, aceleraciones, color=['green', 'orange'])
    plt.title('Aceleración en PyCUDA')
    plt.grid(True)
    for i, v in enumerate(aceleraciones):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')
    plt.savefig('Aceleracion.jpg')
    plt.clf()

    labels = ['Secuencial', 'PyCUDA']
    plt.bar(labels, eficiencias, color=['green', 'orange'])
    plt.title('Eficiencia en PyCUDA')
    plt.grid(True)
    for i, v in enumerate(eficiencias):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom')
    plt.savefig('Eficiencia.jpg')
    plt.clf()"""