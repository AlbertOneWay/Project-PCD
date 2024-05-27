import matplotlib.pyplot as plt
def draw_dotplot(dotplot, img_name='dotplot.jpg'):
    plt.figure(figsize=(10, 10))
    plt.imshow(dotplot, cmap='Greys', interpolation='none')
    plt.title(f'Dotplot de Secuencia 1 vs Secuencia 2')
    plt.xlabel('Secuencia 2')
    plt.ylabel('Secuencia 1')
    plt.savefig(img_name)
    plt.show()
    