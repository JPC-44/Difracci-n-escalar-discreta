import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy.fftpack as fft
import sys


def free_space_transfer_function(fx, fy, Z, λ, retropropagation: bool = False):
    
    k = 2 * np.pi / λ  # wave number

    if retropropagation:
        print("Retropropagación activada")
        return np.exp(-1j * Z * np.sqrt(k**2 - (2 * np.pi * fx)**2 - (2 * np.pi * fy)**2))
    else:
        print("Propagación activada")
        return np.exp(1j * Z * np.sqrt(k**2 - (2 * np.pi * fx)**2 - (2 * np.pi * fy)**2))


def AngularSpectrum(U0, z: float, λ: float, pixel_size: float):

    # Validación de entrada
    if z < N*(pixel_size)**2/λ:                              # condición donde empieza a fallar espectro angular
        print(f'z es mayor que {N*pixel_size**2/λ}')

    # dimensiones del espacio de frecuencias
    M, N = U0.shape

    # espacio
    x = np.arange(-N//2,N//2)
    y = np.arange(-M//2,M//2)
    x,y = x*pixel_size, y*pixel_size
    X,Y = np.meshgrid(x,y)

    # espacio de frecuencias
    fx = np.fft.fftshift(np.fft.fftfreq(N, pixel_size))
    fy = np.fft.fftshift(np.fft.fftfreq(M, pixel_size))
    Fx, Fy = np.meshgrid(fx, fy)

    # función de transferencia en espacio libre H(fx, fy, z)
    transfer_function = free_space_transfer_function(Fx, Fy, z, λ, True)

    # Aplicar un filtro circular para evitar componentes evanescentes
    circ_filter = np.sqrt(fx**2 + fy**2) <= 1 / λ

    # filtering the transfer function
    transfer_function *= circ_filter

    # calculating the optical field at distance Z
    initial_angular_spectrum = np.fft.fftshift(fft.fft2(U0))
    angular_spectrum_at_plane_z = initial_angular_spectrum * transfer_function
    optical_field_at_plane_z = fft.ifft2(np.fft.fftshift(angular_spectrum_at_plane_z))

    return optical_field_at_plane_z



def get_float_input(input_number):
    while True:
        try:
            return float(input(input_number))
        except ValueError:
            print("Por favor, ingrese un número válido.")

def get_int_input(input_number):
    while True:
        try:
            return int(input(input_number))
        except ValueError:
            print("Por favor, ingrese un número entero válido.")
