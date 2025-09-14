import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Parámetros iniciales
z = 0.1  # Distancia entre planos
lamb = 500E-9  # Longitud de onda que se propaga
k = 2 * np.pi / lamb
N = 1024  # Muestras
deltax = 1E-6  # Paso en espacio
deltay = deltax
deltafx = 1 / (N * deltax)
deltafy = deltafx
R = 0.1E-3

# Definición de funciones
def U0(x, y, R):
    return np.where(np.sqrt(x**2 + y**2) < R, 1, 0)

def h(x, y, z):
    return ((np.exp(1j * k * z)) / (1j * lamb * z)) * np.exp(1j * k / (2 * z) * (x**2 + y**2))

# Espacio de coordenadas
x = np.arange(-N // 2, N // 2 + 1) * deltax
y = np.arange(-N // 2, N // 2 + 1) * deltay
X, Y = np.meshgrid(x, y)

# Función para actualizar la gráfica
def update(val):
    z_val = z_slider.val
    R_val = R_slider.val

    Z0 = U0(X, Y, R_val)
    h0 = h(X, Y, z_val)
    H0 = np.fft.fft2(h0)
    F0 = np.fft.fft2(Z0)
    Conv_TF = np.multiply(F0, H0)
    U = np.fft.fftshift(np.fft.ifft2(Conv_TF))

    # Actualizar las imágenes
    ax1.imshow(Z0**2, origin='upper', cmap='gray')
    ax2.imshow(np.abs(U)**2, cmap='gray')
    ax2.set_title(f'Campo propagado z={z_val:.2e} metros.')

    fig.canvas.draw_idle()

# Configuración inicial de la gráfica
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

Z0 = U0(X, Y, R)
h0 = h(X, Y, z)
H0 = np.fft.fft2(h0)
F0 = np.fft.fft2(Z0)
Conv_TF = np.multiply(F0, H0)
U = np.fft.fftshift(np.fft.ifft2(Conv_TF))

img1 = ax1.imshow(Z0**2, origin='upper', cmap='gray')
ax1.set_title('Campo inicial')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

img2 = ax2.imshow(np.abs(U)**2, cmap='gray')
ax2.set_title(f'Campo propagado z={z} metros.')
ax2.set_xlabel('fx')
ax2.set_ylabel('fy')

# Agregar sliders
ax_z = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_R = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')

z_slider = Slider(ax_z, 'z', 0.01, 0.5, valinit=z, valstep=0.01)
R_slider = Slider(ax_R, 'R', 0.01E-3, 0.5E-3, valinit=R, valstep=0.01E-3)

z_slider.on_changed(update)
R_slider.on_changed(update)

plt.show()