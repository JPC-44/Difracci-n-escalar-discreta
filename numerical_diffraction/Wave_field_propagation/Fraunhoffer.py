import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Constantes
wavelength = 500e-9  # Longitud de onda (m)
k = 2 * np.pi / wavelength  # Número de onda
window_size = 4e-3  # Tamaño de la ventana (m)
aperture_radius = 0.2e-3  # Radio de la abertura (m)
grid_points = 500  # Número de puntos en la malla

# Coordenadas en el plano
x = np.linspace(-window_size / 2, window_size / 2, grid_points)
y = np.linspace(-window_size / 2, window_size / 2, grid_points)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Abertura circular
aperture = np.where(R <= aperture_radius, 1, 0)

# Función de propagación de Fraunhofer
def fraunhofer_propagation(z, x_shift, y_shift):
    # Desplazamiento de la abertura en el plano
    shifted_aperture = np.roll(np.roll(aperture, int(x_shift * grid_points / window_size), axis=1),
                               int(y_shift * grid_points / window_size), axis=0)
    U0 = shifted_aperture  # Amplitud inicial
    U_fft = np.fft.fftshift(np.fft.fft2(U0))  # Transformada de Fourier
    intensity = np.abs(U_fft)**2  # Intensidad
    intensity /= intensity.max()  # Normalizar
    return intensity, shifted_aperture

# Configuración inicial
z_initial = 0.1  # Distancia de propagación inicial (m)
x_shift_initial = 0  # Desplazamiento inicial en x (m)
y_shift_initial = 0  # Desplazamiento inicial en y (m)

# Crear figura y ejes
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.25, bottom=0.25)

# Mostrar la abertura inicial
aperture_display = axes[0].imshow(aperture, extent=[-window_size / 2 * 1e3, window_size / 2 * 1e3,
                                                    -window_size / 2 * 1e3, window_size / 2 * 1e3],
                                  cmap='gray', origin='lower')
axes[0].set_title('Abertura Circular')
axes[0].set_xlabel('x (mm)')
axes[0].set_ylabel('y (mm)')

# Mostrar el patrón de Fraunhofer inicial
intensity, shifted_aperture = fraunhofer_propagation(z_initial, x_shift_initial, y_shift_initial)
intensity_display = axes[1].imshow(intensity, extent=[-window_size / 2 * 1e3, window_size / 2 * 1e3,
                                                      -window_size / 2 * 1e3, window_size / 2 * 1e3],
                                   cmap='gray', origin='lower')
axes[1].set_title('Patrón de Difracción (Fraunhofer)')
axes[1].set_xlabel('x (mm)')
axes[1].set_ylabel('y (mm)')

# Crear sliders
ax_z = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_x_shift = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_y_shift = plt.axes([0.25, 0.2, 0.65, 0.03])

slider_z = Slider(ax_z, 'z (m)', 0.05, 1.0, valinit=z_initial)
slider_x_shift = Slider(ax_x_shift, 'x shift (mm)', -2, 2, valinit=x_shift_initial * 1e3)
slider_y_shift = Slider(ax_y_shift, 'y shift (mm)', -2, 2, valinit=y_shift_initial * 1e3)

# Función de actualización para los sliders
def update(val):
    # Obtener los valores de los sliders
    z = slider_z.val
    x_shift = slider_x_shift.val * 1e-3  # Convertir a metros
    y_shift = slider_y_shift.val * 1e-3  # Convertir a metros
    
    # Calcular la propagación de Fraunhofer
    intensity, shifted_aperture = fraunhofer_propagation(z, x_shift, y_shift)
    
    # Actualizar las imágenes
    aperture_display.set_data(shifted_aperture)
    intensity_display.set_data(intensity)
    
    # Redibujar
    fig.canvas.draw_idle()

# Asociar los sliders con la función de actualización
slider_z.on_changed(update)
slider_x_shift.on_changed(update)
slider_y_shift.on_changed(update)

# Mostrar la interfaz
plt.show()

