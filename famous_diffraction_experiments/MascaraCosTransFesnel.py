import numpy as np
import matplotlib.pyplot as plt #condiciones sque funcionan 1024, L=0.4E-3, pixel_size=5E-6,  z= 90E-6
import sys, os
# Subir un nivel desde el archivo actual
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from numerical_diffraction.utils import general_functions as funcs


# Parámetros iniciales
lamb = 633E-6  # longitud de onda que se propaga
N = 1080  # Muestras  #tenia 1024
m = 1  # Valores ideales m=1, L=5E-4, z=8E-4
L = 0.4E-3
z = 0.009E-2  # distancia entre planos
k = 2 * np.pi / lamb
pixel_size = 5E-6  # paso en espacio
output_pixel_size = lamb * z / (N * pixel_size)

print("Tamaño de ventana en la entrada:",N*pixel_size)
print("Pixel en la salida",pixel_size)
print("Tamaño de ventana en la salida",N*output_pixel_size)
print( "Condición de límite de difracción discreta",N * pixel_size**2 / lamb)


def U0(m: int, L: float):

    x_0 = np.arange(-N//2,N//2) * pixel_size
    y_0 = np.arange(-N//2, N//2) * pixel_size
    X_0, Y_0 = np.meshgrid(x_0,y_0)

    return 0.5 * (1 + m * np.cos(2 * np.pi * X_0 / L + 0 * Y_0))

# Campo óptico que se calculó a mano 
def campo_optico_analitico(z, output_pixel_size):
    
    x = np.arange(-N//2,N//2)*output_pixel_size
    y = np.arange(-N//2,N//2)*output_pixel_size
    X,Y = np.meshgrid(x,y)

    faseCalculada = np.exp(-1j * np.pi * lamb * z / (L**2))
    return (1 / 2) * (1 + faseCalculada * m * np.cos(2 * np.pi * X / L + 0 * Y))


#Funcion main donde suceden cositas
def main():
    
    entrance_optical_field = U0(m, L)

    optical_field_plane_z, coordenadas = funcs.Fresnel_transformation(entrance_optical_field, z, lamb, pixel_size)
    intensity = (np.abs(optical_field_plane_z))**2                         # Intensidad de Campo calculado computacionalmente

    # Cálculos adicionales
    campo_analitico = campo_optico_analitico(z, output_pixel_size)   
    intensidad_campo_analitico  = np.abs(campo_analitico) **2 
    intensidad_campo_entrada = (np.abs(entrance_optical_field))**2

    # Perfil
    perfil_horizontal = intensity[N // 2, :]

    # Normalizacion del perfil
    perfil_horizontal = perfil_horizontal/np.max(perfil_horizontal)

    # Graficas
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(intensidad_campo_entrada, cmap='gray', extent=(min(coordenadas[0]), max(coordenadas[0]), min(coordenadas[1]), max(coordenadas[1])), origin='upper')
    plt.colorbar()
    plt.title('Distribución de energía en el plano de entrada')
    plt.xlabel('x0 (m)')
    plt.ylabel('y0 (m)')

    plt.subplot(2, 2, 2)
    plt.imshow(intensity, extent=(min(coordenadas[2]), max(coordenadas[2]), min(coordenadas[3]), max(coordenadas[3])), origin='upper', cmap='gray')
    plt.colorbar()
    plt.title(f'Distribución de energía en el plano z0={z} m')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    plt.subplot(2, 2, 3)
    plt.imshow(intensidad_campo_analitico, extent=(min(coordenadas[2]), max(coordenadas[2]), min(coordenadas[3]), max(coordenadas[3])), origin='upper', cmap='gray')
    plt.colorbar()
    plt.title(f'Distribución de energía (Cálculo analítico) z0={z} m')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    Perfil=intensidad_campo_analitico[N // 2, :]
    Perfil=Perfil/np.max(Perfil)
    # Plot 
    plt.subplot(2, 2, 4)
    plt.plot(coordenadas[2], Perfil, label='Intensidad Analítica')
    plt.plot(coordenadas[2], perfil_horizontal, label='Perfil Horizontal (Computacional)', linestyle='--')
    plt.title("Intensidad del Campo Difractado")
    plt.xlabel("x (m)")
    plt.ylabel("Intensidad")
    plt.legend(fontsize=7)  # Ajusta el tamaño de la fuente de la leyenda
    plt.grid()
    plt.show()

if z < N * pixel_size**2 / lamb:
    main()
else:
    main()
    print(f'z es mayor que {N * pixel_size**2 / lamb}')

