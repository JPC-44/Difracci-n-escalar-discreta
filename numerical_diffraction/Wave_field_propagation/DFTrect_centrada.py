import numpy as np
import matplotlib.pyplot as plt

# Crear una señal 2D de ejemplo (imagen con un cuadrado en el centro)
N = 64  # Tamaño de la señal en ambas dimensiones
signal = np.zeros((N, N))
signal[N//4:3*N//4, N//4:3*N//4] = 1  # Cuadrado blanco en el centro

# Calcular la FFT 2D de la señal
dft_2d = np.fft.fft2(signal)

# Sin aplicar el desplazamiento
magnitude_spectrum = np.abs(dft_2d)

# Aplicar el desplazamiento en la FFT 2D
dft_2d_shifted = np.fft.fftshift(dft_2d)
magnitude_spectrum_shifted = np.abs(dft_2d_shifted)

# Visualizar la señal original y su FFT 2D con y sin centrar
plt.figure(figsize=(18, 6))

# Señal original
plt.subplot(1, 3, 1)
plt.imshow(signal, cmap='gray')
plt.title('Señal original')
plt.colorbar(label='Intensidad')
plt.xlabel('Posición X')
plt.ylabel('Posición Y')

# FFT 2D sin centrar
plt.subplot(1, 3, 2)
plt.imshow(np.log(magnitude_spectrum + 1), cmap='gray')
plt.title('FFT 2D sin centrar')
plt.colorbar(label='Magnitud (log scale)')
plt.xlabel('Frecuencia horizontal')
plt.ylabel('Frecuencia vertical')

# FFT 2D centrada
plt.subplot(1, 3, 3)
plt.imshow(np.log(magnitude_spectrum_shifted + 1), cmap='gray', extent=(-np.pi, np.pi, -np.pi, np.pi))
plt.title('FFT 2D centrada')
plt.colorbar(label='Magnitud (log scale)')
plt.xlabel('Frecuencia horizontal')
plt.ylabel('Frecuencia vertical')

plt.show()