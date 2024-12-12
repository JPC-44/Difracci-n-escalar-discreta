import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Carga de la imagen y conversión a escala de grises
image = Image.open('Logo_OD.png').convert('L')  # Convertir a escala de grises
Z0 = np.sqrt(np.array(image, dtype=np.float64))  # Calcular la raíz cuadrada de la intensidad

# Parámetros iniciales
z = 0.4              # Distancia entre planos
lamb = 650E-9           # Longitud de onda que se propaga
k = 2 * np.pi / lamb    # Número de onda
M, N = Z0.shape         # Dimensiones de la imagen
deltax = 1E-6           # Paso en espacio
deltay = deltax         # Igual a deltax para simplificar
deltafx = lamb*z / (N * deltax)  # Resolución en frecuencia espacial
deltafy = lamb*z / (M * deltay)  # Resolución en frecuencia espacial



# Definición de la función de propagación
def h(x, y, z):
    return ((np.exp(1j * k * z)) / (1j * lamb * z)) * np.exp(1j * k / (2 * z) * (x**2 + y**2))


np.exp((1j * k / 2*z)*((deltafx)))


# Espacio de coordenadas
x = np.linspace(-N // 2, N // 2 - 1, N) * deltax
y = np.linspace(-M // 2, M // 2 - 1, M) * deltay
X, Y = np.meshgrid(x, y)

# Cálculo de la función de transferencia
H0 = h(X, Y, z)
#H0 = np.fft.fft2(h0)  # Transformada de Fourier de la función de propagación
F0 = np.fft.fft2(Z0)  # Transformada de Fourier de la intensidad inicial
Conv_TF = np.multiply(F0, H0)  # Producto en frecuencia
U = np.fft.fftshift(np.fft.ifft2(Conv_TF))  # Resultado propagado, centrado

# Configuración inicial de la gráfica
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Gráfica del campo inicial
img1 = ax1.imshow(Z0**2, origin='upper', cmap='gray')
ax1.set_title('Campo inicial')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Gráfica del campo propagado
img2 = ax2.imshow(np.abs(U)**2, cmap='gray')
ax2.set_title(f'Campo propagado z={z} metros')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.show()
