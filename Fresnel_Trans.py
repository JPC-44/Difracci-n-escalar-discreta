import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

# Carga de la imagen y conversión a escala de grises
image = Image.open('Logo_OD.png').convert('L')  # Convertir a escala de grises
Z0 = np.sqrt(np.array(image, dtype=np.float64))  # Calcular la raíz cuadrada de la intensidad


z= 0.2     #distancia entre planos
lamb=650E-9                            #longitud de onda que se propaga
k=2*np.pi/lamb
M, N = Z0.shape         # Dimensiones de la imagen
deltax_0 = 10E-6                         #paso en espacio
deltax = lamb*z/(N*deltax_0)



#Fase parabolica en el plano Z=0
def Fase2(x_0,y_0):
    return np.exp((1j*k/(2*z))*(x_0**2+y_0**2))

# Fase parabólica en el plano de salida. 
def ConstantPhase(x,y):
    return (np.exp(1j*k*z))*(np.exp((1j*k/(2*z))*(x**2+y**2)))/(1j*lamb*z)

# Producto entre fase parabólica por el la transmitancia en z=0
def U_1(x_0,y_0):
    Product=(deltax**2)*Fase2(x_0,y_0)*Z0
    return Product

def main():

    # Marca de tiempo inicial
    start_time = time.perf_counter()
    print("Inicio del proceso...")


    x=np.arange(-N//2,N//2)
    y=np.arange(-N//2,N//2)
    x,y=x*deltax,y*deltax               #Darle dimensiones al plano z=0
    X,Y=np.meshgrid(x,y)

    x_0=np.arange(-N//2,N//2)
    y_0=np.arange(-N//2,N//2)       
    x_0,y_0=x_0*deltax_0,y_0*deltax_0   #Darle dimensiones al plano z=z
    X_0,Y_0=np.meshgrid(x_0,y_0)

    U1=U_1(X_0,Y_0)                     #Evaluar  U_1 en corrdenadas del plano z=0

    F0=np.fft.fftshift(np.fft.fft2(U1))     #Se le aplica la transformada de Fourier y se centran las frecuencias
    F1=np.multiply(F0,ConstantPhase(X,Y))   #Se multiplica punto  a punto la transformada de Fourier y la fase que res constante respecto a


    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Tiempo transcurrido transformada de Fresnel con FFT: {elapsed_time:.8f} segundos")
    
    ##'''Creación de figura, plano z=0 y plano difractado'''

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(U1)**2, extent=(min(x_0), max(x_0), min(y_0), max(y_0)), origin='upper', cmap='gray')
    plt.colorbar()
    plt.title('Distribución de energía en el plano de entrada')
    plt.xlabel('x0')
    plt.ylabel('y0')

    plt.subplot(1, 2, 2)
    plt.imshow((((np.abs(F1))**2)), extent=(min(x), max(x), min(x), max(x)), origin='upper', cmap='gray')
    plt.colorbar()
    plt.title(f'Distribución de energía en el plano z0={z} m.')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.show()
    return 0

if z<N*deltax**2/lamb:
    main()
else:
    main()
    print(f'z es mayor que {N*deltax**2/lamb}')