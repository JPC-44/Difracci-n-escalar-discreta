import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy.fftpack as fft
import os

# Importar máscara para Z=0


#image = Image.open("C:/Users/Usuario/Desktop/LogoODP_propagado/image_4.tif").convert('L')  # Convertir a escala de grises

image2= Image.open("C:/Users/Usuario/Desktop/GitHub/Python-Programs/DMD/images_to_display/transmitancia9/image_stack.tif").convert('L')  # Convertir a escala de grises
image2 = np.array(image2, dtype=np.float64)

try:
    image3 = Image.open('Imag.png').convert('L')  # Convertir a escala de grises
    image3 = np.array(image3, dtype=np.float64)
except FileNotFoundError:
    print("No se encontró 'Imag.png'. Usando matriz de ceros para la parte imaginaria.")
    image3 = np.zeros_like(image2, dtype=np.float64)    

Campo = image2 + 1j * image3

λ = 633e-9  # Longitud de onda en metros
pixel = 5.2E-6  # Tamaño de píxel en metros

#U0 = np.array(image, dtype=np.float64)  # Convertir imagen a float64 para mayor precisión
#U0 = np.array(Campo, dtype=np.complex128)  # Usar tipo complejo
U0 = np.array(np.uint8(Campo))

# Crear carpeta para guardar las imágenes
output_folder = "retropropagacion/iteracion_1"
os.makedirs(output_folder, exist_ok=True)

# Generar imágenes y guardarlas

""" UZ_magnitude = (np.abs(U0))**2

U0= np.abs(U0)**2 """



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
stop_Z = get_float_input("Ingrese el valor final de Z (en metros): ")

num_images = int((stop_Z - start_Z) / step_Z) + 1

print(f"Se generarán {num_images} imágenes.")

# Crear carpeta para guardar las imágenes



# Generar imágenes y guardarlas
for i in range(num_images):
    Z = start_Z + i * step_Z
    UZ = AngularSpectrum(U0, Z, λ, pixel)
    UZ_magnitude = np.abs(UZ)**2

    output_path = os.path.join(output_folder, f"propagated_Z_{Z:.5f}m.png")
    plt.imsave(output_path, UZ_magnitude, cmap='gray')
    print(f"Imagen guardada: {output_path}")

print(f"Imágenes generadas y guardadas en la carpeta: {output_folder}")
