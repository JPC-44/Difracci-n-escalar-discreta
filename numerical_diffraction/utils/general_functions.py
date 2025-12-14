import numpy as np
import scipy.fftpack as fft
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os


""""insertar una clase"""""
def free_space_transfer_function(fx, fy, Z, λ, retropropagation: bool = False):
    
    k = 2 * np.pi / λ  # wave number

    if retropropagation:
        return np.exp(-1j * Z * np.sqrt(k**2 - (2 * np.pi * fx)**2 - (2 * np.pi * fy)**2))
    else:
        return np.exp(1j * Z * np.sqrt(k**2 - (2 * np.pi * fx)**2 - (2 * np.pi * fy)**2))

"""Insertar una clase para funciones de transferencia"""""
def Fresnel_transfer_function(x, y, z, λ, retropropagation: bool = False):
    
    k = 2 * np.pi / λ  # wave number

    if retropropagation == True:
        constante_phase = (np.exp(1j * k * (-1*z))) / (1j * λ * (-1*z))
        fourier_transform_psf =  np.fft.fft2( np.exp(1j * k / (2 * (-1*z)) * (x**2 + y**2)) )
        transfer_function = constante_phase * fourier_transform_psf
    
    else:
        constante_phase = (np.exp(1j * k * z)) / (1j * λ * z)
        fourier_transform_psf =  np.fft.fft2( np.exp(1j * k / (2 * z) * (x**2 + y**2)) )
        transfer_function = constante_phase * fourier_transform_psf

    return transfer_function    

"""computation of the angular spectrum method"""
def AngularSpectrum(U0, z: float, λ: float, pixel_size: float, retropropagation = False):
    """
    U0: complex array to propagate
    z: distance to propagate [m]
    λ: wavelenght[m] of the optical wave field
    input_pixel_size [m]: is the size of the pixel in the plane z=0
    retropropagation: defines if the method will go in z+ or z-
    """

    # dimensiones del espacio de frecuencias
    M, N = U0.shape

    # Validación de entrada
    if z < N*(pixel_size)**2/λ:                              # condición donde empieza a fallar espectro angular
        print(f'z es mayor que {N*pixel_size**2/λ}')

    # espacio
    x = np.arange(-N//2,N//2)
    y = np.arange(-M//2, M//2)
    x,y = x*pixel_size, y*pixel_size
    X,Y = np.meshgrid(x,y)

    # espacio de frecuencias
    fx = np.fft.fftshift(np.fft.fftfreq(N, pixel_size))
    fy = np.fft.fftshift(np.fft.fftfreq(M, pixel_size))
    #al evaluar en este meshgrid que está hecho con np.fftfreq salen las coordenadas centradas 
    # dentro de la evaluación de una función. Por lo que al multiplica por una transformada
    # de Fourier toca realizar un proceso de shifting primero al espectro para tener congruencia en los resultados
    Fx, Fy = np.meshgrid(fx, fy)

    # función de transferencia en espacio libre H(fx, fy, z)
    transfer_function = free_space_transfer_function(Fx, Fy, z, λ, retropropagation)
    
    # Aplicar un filtro circular para evitar componentes evanescentes --> poco importante debido a que nunca se llegará a esas frecuencias 10^9 m^-1
    circ_filter = np.sqrt(Fx**2 + Fy**2) <= 1 / λ

    # filtering the transfer function
    transfer_function = transfer_function * circ_filter

    # calculating the optical field at distance Z
    initial_angular_spectrum = np.fft.fftshift(fft.fft2(U0))
    angular_spectrum_at_plane_z = initial_angular_spectrum * transfer_function
    optical_field_at_plane_z = fft.ifft2(np.fft.fftshift(angular_spectrum_at_plane_z))

    return optical_field_at_plane_z

"""computation of Fresnel transformation method"""
def Fresnel_transformation(U_0, z: float, λ: float, input_pixel_size: float):

    """
    U0: complex array to propagate
    z: distance to propagate [m]
    λ: wavelenght [m] of the optical wave field
    input_pixel_size [m]: is the size of the pixel in the plane z=0
    """
    M, N = U_0.shape
    output_pixel_size = λ * z /(N * input_pixel_size)
    k = 2 * np.pi / λ  # wave number

    # Dimensiones en plano z=0
    x_0 = np.arange(-N//2,N//2)
    y_0 = np.arange(-N//2,N//2)       
    x_0,y_0 = x_0*input_pixel_size, y_0*input_pixel_size   
    X_0,Y_0 = np.meshgrid(x_0,y_0)

    # dimensiones en plano z=z
    x = np.arange(-N//2,N//2)
    y = np.arange(-M//2,M//2)
    x,y = x*output_pixel_size, y*output_pixel_size
    X,Y = np.meshgrid(x,y)


    # producto entre fase parabólica y la transmitancia en z=0
    parabolic_phase_0 = (np.exp(    (1j*k/(2*z)) * (X_0**2 + Y_0**2))) * (input_pixel_size**2)
    constant_phase_output_plane = (np.exp(1j*k*z))*(np.exp((1j*k/(2*z))*(X**2+Y**2)))/(1j*λ*z)

    product = np.multiply(parabolic_phase_0, U_0)

    fourier_transform_of_product = np.fft.fftshift(np.fft.fft2(product))
    
    optical_wave_field = np.multiply(fourier_transform_of_product, constant_phase_output_plane)

    if z<N*input_pixel_size**2/λ:
        print(f'z es menor que {N*input_pixel_size**2/λ}')
        print("Se recomienda usar el método de espectro angular para esta distancia.")
    else:
        print(f'z es mayor que {N*input_pixel_size**2/λ}')
        print("Método de Fresnel aplicado correctamente.")
    
    array_coordenadas = [x_0, y_0, x, y]


    return optical_wave_field, array_coordenadas

"""computation of the Fresnel convolution method"""
def Fresnel_convolution(U0, z: float, λ: float, pixel_size: float, retropropagation = False):
    M, N = U0.shape

    # espacio z=z
    x = np.arange(-N//2,N//2)
    y = np.arange(-M//2,M//2)
    x,y = x*pixel_size, y*pixel_size
    X,Y = np.meshgrid(x,y)
   

    # transformada de Fourier del campo inicial
    fourier_transform_entrace_field = np.fft.fft2(U0)

    # función de transferencia en espacio libre 
    #  bajo aproximación paraxial H(fx, fy, z)
    transfer_function = Fresnel_transfer_function(X, Y, z, λ, retropropagation)

    # producto en frecuencia
    fresnel_convolution = np.multiply(fourier_transform_entrace_field, transfer_function)

    # resultado propagado, centrado
    optical_field_at_z = np.fft.fftshift(np.fft.ifft2(fresnel_convolution))

    if z < N*pixel_size**2/λ:
        print(f'z es menor que {N*pixel_size**2/λ}')
        print("Se recomienda usar el método de espectro angular para esta distancia.")
    else:
        print(f'z es mayor que {N*pixel_size**2/λ}')
        print("Método de Fresnel por convolución aplicado correctamente.")
    
    return optical_field_at_z


""" Aquí insertar una clase para manejar las entradas de usuario y evitar repetición de código """
def load_complex_field(real_path: Path, imag_path: Path) -> np.ndarray:
    """
    Carga dos imágenes (parte real e imaginaria) y devuelve un campo complejo.
    """
    real = cv2.imread(str(real_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    imag = cv2.imread(str(imag_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return real + 1j * imag

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

def cargar_imagen(path):

    """Carga una imagen en escala de grises como float64 normalizado."""
    try:
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float64)
    except FileNotFoundError:
        print(f"No se encontró la imagen: {path}")
        return None
    
def crear_campo(real_img = None, imag_img = None, intensity = None):
    """Combina parte real e imaginaria en un campo complejo."""
    
    if real_img is None and imag_img is None and intensity is not None:
        print("Campo formado con imagen de intensidad")
        return intensity
    
    elif real_img is None and imag_img is not None:
        print("Campo complejo = parte real")
        return 1j * imag_img
        
    elif imag_img is None and real_img is not None:
        print("Campo complejo = parte imaginaria")
        return real_img
    
    elif imag_img is not None and real_img is not None:
        print("Campo complejo = parte imag + real")
        return real_img + 1j * imag_img


"""visualización con slider
   de un stack de imagenes"""
def slice_images(imagenes, labels=None):

    if labels == None:
        labels = [str(i) for i in range(len(imagenes))]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Mostrar primera imagen
    img_plot = ax.imshow(imagenes[0], cmap='gray')
    ax.set_title("Imagen 0")
    ax.axis('off')

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Index', 0, len(imagenes)-1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        img_plot.set_data(imagenes[idx])
        ax.set_title(f"distance {labels[idx]}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


"""
Guardar imagenes dado un array de imágenes,
el label para nombrar la imágen y el path
"""
def guardar_imagenes(stack, label, path):
    for i in range(0,len(stack)):

        output_path = os.path.join(path, f"z_{label[i]}m.png")
        plt.imsave(output_path, stack[i], cmap='gray')
        print(f"Imagen de intensidad guardada: {output_path}")

    print(f"Imágenes generadas y guardadas en la carpeta: {output_path}")

