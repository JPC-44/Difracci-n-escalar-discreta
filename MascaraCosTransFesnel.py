import numpy as np
import matplotlib.pyplot as plt #condiciones sque funcionan 1024, L=0.4E-3, deltax_0=5E-6,  z= 90E-6

# Parámetros iniciales
lamb = 633E-6  # longitud de onda que se propaga
N = 1024  # Muestras  #tenia 1024
m = 1  # Valores ideales m=1, L=5E-4, z=8E-4
L = 0.4E-3
z = 0.009E-2  # distancia entre planos
#z = N*(L**2)/lamb
k = 2 * np.pi / lamb


deltax_0 = 5E-6  # paso en espacio
deltax = lamb * z / (N * deltax_0)
print("Tamaño de ventana en la entrada:",N*deltax_0)
print("Pixel en la salida",deltax_0)
print("Tamaño de ventana en la salida",N*deltax)
print( "Condición de límite de difracción discreta",N * deltax**2 / lamb)
# Función t, mascara
def U0(x, y):             
    return 0.5 * (1 + m * np.cos(2 * np.pi * x / L+0*y))

#Fase parabólica que está en coordenadas del plano de entrada
def Fase2(x_0, y_0):
    return np.exp((1j * k / (2 * z)) * (x_0**2 + y_0**2))

# Fase constatne. Se le multiplica a la transformada de Fresnel al final de la fft
def ConstantPhase(x, y):
    return (np.exp(1j * k * z)) * (np.exp((1j * k / (2 * z)) * (x**2 + y**2))) / (1j * lamb * z)

# Producto entre la mascara y la fase en el plano z=0 de manera que sea la funcion que ingresa a la fft
def U_1(x_0, y_0):
    return Fase2(x_0, y_0) * U0(x_0, y_0)

# Campo óptico que se calculó a mano 
def Campo_Optico_analitico(x, y, z):
    faseCalculada = np.exp(-1j * np.pi * lamb * z / (L**2))
    return (1 / 2) * (1 + faseCalculada * m * np.cos(2 * np.pi * x / L))


#Funcion main donde suceden cositas
def main():
    x = np.arange(-N // 2, N // 2) * deltax
    y = np.arange(-N // 2, N // 2) * deltax                 #Meshgrid para el plano de salida
    X, Y = np.meshgrid(x, y)

    x_0 = np.arange(-N // 2, N // 2) * deltax_0
    y_0 = np.arange(-N // 2, N // 2) * deltax_0             #Mesgrid para plano de entrada
    X_0, Y_0 = np.meshgrid(x_0, y_0)

    U1 = U_1(X_0, Y_0)                                      # Campo óptico en z=0
    F0 = np.fft.fftshift(np.fft.fft2(U1))                   #Transformada de fourier con centrado de freq.
    F1 = np.multiply(F0, ConstantPhase(X, Y))               # Multiplicación 

    intCampoCompu = (np.abs(F1))**2                         # Intensidad de Campo calculado computacionalmente
    intCampoCompu = intCampoCompu / np.max(intCampoCompu)  # Normalización

    # Cálculos adicionales
    Campo = Campo_Optico_analitico(X, Y, z)                 # Campo optico calculado analiticamente.
    intCampoAnalitico = (np.abs(Campo))**2 / (np.max(np.abs(Campo))**2)  # Normalización
    intCampoEntrada = (np.abs(U1))**2 / np.max(np.abs(U1))  # Normalización

    # Perfil
    perfil_horizontal = intCampoCompu[N // 2, :]

    # Normalizacion del perfil
    perfil_horizontal = perfil_horizontal/np.max(perfil_horizontal)

    # Graficas
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(intCampoEntrada, extent=(min(x_0), max(x_0), min(y_0), max(y_0)), origin='upper', cmap='gray')
    plt.colorbar()
    plt.title('Distribución de energía en el plano de entrada')
    plt.xlabel('x0 (m)')
    plt.ylabel('y0 (m)')

    plt.subplot(2, 2, 2)
    plt.imshow(intCampoCompu, extent=(min(x), max(x), min(y), max(y)), origin='upper', cmap='gray')
    plt.colorbar()
    plt.title(f'Distribución de energía en el plano z0={z} m')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    plt.subplot(2, 2, 3)
    plt.imshow(intCampoAnalitico, extent=(min(x), max(x), min(y), max(y)), origin='upper', cmap='gray')
    plt.colorbar()
    plt.title(f'Distribución de energía (Cálculo analítico) z0={z} m')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    Perfil=intCampoAnalitico[N // 2, :]
    Perfil=Perfil/np.max(Perfil)
    # Plot 
    plt.subplot(2, 2, 4)
    plt.plot(x, Perfil, label='Intensidad Analítica')
    plt.plot(x, perfil_horizontal, label='Perfil Horizontal (Computacional)', linestyle='--')
    plt.title("Intensidad del Campo Difractado")
    plt.xlabel("x (m)")
    plt.ylabel("Intensidad")
    plt.legend(fontsize=7)  # Ajusta el tamaño de la fuente de la leyenda
    plt.grid()
    plt.show()

if z < N * deltax**2 / lamb:
    main()
else:
    main()
    print(f'z es mayor que {N * deltax**2 / lamb}')






#Siempre directivas de preprocesamiento #includes
# Variables u objetos que serán variables globales
# 'prototipos de las funciones'declaración de funciones
# int main(void)
# se configura todo en el main, gpio,timers, exti
# siempre debe existir el ciclo infinito en el main

#luego se siguen las funciones que se definieron arriba en los prototipos (cuerpo de las funciones)
#al final están los callbacks
# realizar los 15 cases en el exti
