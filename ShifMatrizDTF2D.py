import numpy as np
import matplotlib.pyplot as plt

def circ(x, y):
    """
    Define una función de circunferencia en 2D.
    
    Args:
        x (numpy.ndarray): Coordenadas x de la malla.
        y (numpy.ndarray): Coordenadas y de la malla.
        
    Returns:
        numpy.ndarray: La función de circunferencia.
    """
    return np.where(x**2 + y**2 < 1, 1, 0)

def fft2d_with_shift(signal_func, N):
    """
    Calcula la Transformada de Fourier 2D (FFT) de una señal y aplica el desplazamiento (shift) necesario.
    
    Args:
        signal_func (function): Función que representa la señal 2D.
        N (int): Tamaño de la señal en ambas dimensiones (N x N).
        
    Returns:
        numpy.ndarray: La FFT 2D de la señal con el desplazamiento aplicado.
    """
    # Crear una malla de coordenadas
    x = np.linspace(-np.pi, np.pi, N)
    y = np.linspace(-np.pi, np.pi, N)
    X, Y = np.meshgrid(x, y)
    
    # Calcular la señal 2D utilizando la función proporcionada
    signal = signal_func(X, Y)
    
    # Calcular la FFT 2D de la señal
    dft_2d = np.fft.fft2(signal)
    
    # Aplicar el desplazamiento en la FFT 2D
    dft_2d_shifted = np.fft.fftshift(dft_2d)
    
    return dft_2d_shifted

# Tamaño de la señal en ambas dimensiones
N = 64

# Calcular la FFT 2D de la señal con el desplazamiento
fft_2d_shifted = fft2d_with_shift(circ, N)

# Visualizar la magnitud de la FFT 2D desplazada
plt.figure(figsize=(8, 8))
plt.imshow(np.abs(fft_2d_shifted), cmap='gray', extent=(-np.pi, np.pi, -np.pi, np.pi))
plt.title('FFT 2D de una función de circunferencia desplazada')
plt.colorbar(label='Magnitud')
plt.xlabel('Frecuencia horizontal')
plt.ylabel('Frecuencia vertical')
plt.show()