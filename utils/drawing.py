import matplotlib.pyplot as plt
def draw_dotplot(dotplot, img_name='dotplot.jpg'):
    plt.figure(figsize=(10, 10))
    plt.imshow(dotplot, cmap='Greys', interpolation='none')
    plt.title(f'Dotplot de Secuencia 1 vs Secuencia 2')
    plt.xlabel('Secuencia 2')
    plt.ylabel('Secuencia 1')
    plt.savefig(img_name)
    plt.show()

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