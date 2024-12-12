import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

m=0.5            
L=100E-4
z=10E-2                         #distancia entre planos
lamb=633E-6                     #longitud de onda que se propaga
k=2*np.pi/lamb
N=1024                          #  Muestras      NOTA: tomar muestras proporcionales a C, para que el pixel se tome bien en la matriz C=1/deltax
deltax_0 = 100E-6                 #paso en espacio
#z=N*(L**2)/lamb
deltax = lamb*z/(N*deltax_0)
dimensionesImagen=N*deltax
dimensionesImagenPropagada=N*deltax_0
print("distancia de propagación:",z)
print("Dimensiones de la imagen en:",dimensionesImagen)
print("Dimensiones imagen en z:",dimensionesImagenPropagada)





        # Máscara sinusoidal
def U0(x,y):
    return 0.5*( 1 + m*np.cos(2*np.pi*x/L+0*y))   

def Campo_Optico_analitico(x,y,z):

    faseCalculada=np.exp(-1j*np.pi*lamb*z/(L**2))
    campoOptico=(1/2)*( 1  + faseCalculada*m*np.cos(2*np.pi*x/L+0*y))
    return campoOptico

#Fase parabolica en el plano Z=0
def Fase2(x_0,y_0):
    return np.exp((1j*k/(2*z))*(x_0**2+y_0**2))

# Fase parabólica en el plano de salida. 
def ConstantPhase(x,y):
    return (np.exp(1j*k*z))*(np.exp((1j*k/(2*z))*(x**2+y**2)))/(1j*lamb*z)

# Producto entre fase parabólica por el la transmitancia en z=0
def U_1(x_0,y_0):
    Product=Fase2(x_0,y_0)*U0(x_0,y_0)
    return Product

def main():
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
                                            # la transformada de Fourier
    

    Campo=Campo_Optico_analitico(X,Y,z)
    intCampoAnalitico=(np.abs(Campo))**2
    intCampoCompu=(np.abs(F1))**2
    intCampoEntrada=(np.abs(U1))**2
    ##'''Creación de figura, plano z=0 y plano difractado'''
    intCampoCompu=intCampoCompu/np.max(intCampoCompu)
    print(np.max(intCampoCompu))
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title('Distribución de energía en el plano de entrada')
    plt.imshow(intCampoEntrada, extent=(min(x_0), max(x_0), min(y_0), max(y_0)), origin='upper', cmap='gray')
    plt.colorbar()
    plt.title('Distribución de energía en el plano de entrada')
    plt.xlabel('x0')
    plt.ylabel('y0')

    plt.subplot(2, 2, 2)
    plt.imshow(intCampoCompu, extent=(min(x), max(x), min(x), max(x)), origin='upper', cmap='gray',vmax=0.1)
    plt.colorbar()
    plt.title(f'Distribución de energía en el plano z0={z} m.')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(2, 2, 3)
    plt.imshow(intCampoAnalitico, extent=(min(x), max(x), min(x), max(x)), origin='upper', cmap='gray')
    plt.colorbar()
    plt.title(f'Distribución de energía en el plano z0={z} m. Cálculo análitico')
    plt.xlabel('x')
    plt.ylabel('y')



    term1 = 1
    term2 = m**2 * np.cos(2 * np.pi * x / L)**2
    term3 = m * np.cos(2 * np.pi * x / L) * np.cos(np.pi * lamb * z / L**2)
    intensidad = (1/4) * (term1 + term2 + term3)

    # plot 1d
    plt.figure(figsize=(8, 4))
    plt.plot(x, intensidad,label='')
    plt.title("Intensidad del Campo Difractado")
    plt.xlabel(r"$x$")
    plt.ylabel(r"")
    plt.grid()
    plt.show()
    return 0

if z<N*deltax**2/lamb:
    main()
else:
    main()
    print(f'z es mayor que {N*deltax**2/lamb}')



z=N*L**2/lamb

deltax = lamb*z/(N*deltax_0)
z<N*deltax**2/lamb