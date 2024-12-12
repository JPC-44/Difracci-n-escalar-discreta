import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy.fftpack as fft
import os

# Función para agregar padding
def add_padding(image, factor=4):
    """Agrega padding para cuadruplicar el tamaño de la imagen en cada eje."""
    original_shape = image.shape
    padded_shape = (original_shape[0] * factor, original_shape[1] * factor)
    padded_image = np.zeros(padded_shape, dtype=image.dtype)

    start_row = (padded_shape[0] - original_shape[0]) // 2
    start_col = (padded_shape[1] - original_shape[1]) // 2

    padded_image[start_row:start_row + original_shape[0], start_col:start_col + original_shape[1]] = image

    return padded_image

# Función para remover padding
def remove_padding(image, original_shape):
    """Elimina el padding de una imagen para restaurar su tamaño original."""
    padded_shape = image.shape

    start_row = (padded_shape[0] - original_shape[0]) // 2
    start_col = (padded_shape[1] - original_shape[1]) // 2

    return image[start_row:start_row + original_shape[0], start_col:start_col + original_shape[1]]

# Importar máscara para Z=0
image = Image.open('Intensity.png').convert('L')  # Convertir a escala de grises

image2 = Image.open('Real.png').convert('L')  # Convertir a escala de grises
image2 = np.array(image2, dtype=np.float64)

image3 = Image.open('Imag.png').convert('L')  # Convertir a escala de grises
image3 = np.array(image3, dtype=np.float64)

Campo = image2 + 1j * image3

λ = 633e-9  # Longitud de onda en metros
pixel = 3.45e-6  # Tamaño de píxel en metros

U0 = np.sqrt(np.array(image, dtype=np.float64))  # Convertir imagen a float64 para mayor precisión
#U0 = np.array(Campo, dtype=np.complex128)  # Usar tipo complejo


""""
# Agregar padding
original_shape = U0.shape
U0_padded = add_padding(U0)

# Crear carpeta para guardar las imágenes
output_folder = "imagenes_propagadas"
os.makedirs(output_folder, exist_ok=True)


# Generar imágenes y guardarlas
U0_magnitude = np.abs(U0_padded)**2

output_path = os.path.join(output_folder, f"IntensityCalculated.png")
plt.imsave(output_path, U0_magnitude, cmap='gray')
print(f"Imagen guardada: {output_path}")

print(f"Imágenes generadas y guardadas en la carpeta: {output_folder}")

"""


def AngularSpectrum(U0, Z, λ, pixel):
    M, N = U0.shape
    fx = np.fft.fftshift(np.fft.fftfreq(N, pixel))
    fy = np.fft.fftshift(np.fft.fftfreq(M, pixel))
    fx, fy = np.meshgrid(fx, fy)

    K = 2 * np.pi / λ
    H = np.exp(-1j * Z * np.sqrt(K**2 - (2 * np.pi * fx)**2 - (2 * np.pi * fy)**2))

    mask = np.sqrt(fx**2 + fy**2) <= 1 / λ
    H *= mask

    A0 = np.fft.fftshift(fft.fft2(U0))
    Az = A0 * H
    Uz = fft.ifft2(np.fft.fftshift(Az))

    return Uz

def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Por favor, ingrese un número válido.")

def get_int_input(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Por favor, ingrese un número entero válido.")

# Parámetros ajustables
start_Z = get_float_input("Ingrese el valor inicial de Z (en metros): ")
step_Z = get_float_input("Ingrese el paso entre valores de Z (en metros): ")
num_images = get_int_input("Ingrese la cantidad de imágenes a generar: ")

# Crear carpeta para guardar las imágenes
output_folder = "imagenes_propagadas"
os.makedirs(output_folder, exist_ok=True)



# Generar imágenes y guardarlas
for i in range(num_images):
    Z = start_Z + i * step_Z
    UZ = AngularSpectrum(U0, Z, λ, pixel)

    # Remover padding
    
    UZ_magnitude = np.abs(UZ)**2

    output_path = os.path.join(output_folder, f"propagated_Z_{Z:.5f}m.png")
    plt.imsave(output_path, UZ_magnitude, cmap='gray')
    print(f"Imagen guardada: {output_path}")

print(f"Imágenes generadas y guardadas en la carpeta: {output_folder}")
