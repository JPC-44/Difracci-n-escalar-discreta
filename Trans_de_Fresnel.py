import numpy as np
import matplotlib.pyplot as plt


z=30E-2     #distancia entre planos
lamb=500E-9                            #longitud de onda que se propaga
k=2*np.pi/lamb
N=1024       #  Muestras      NOTA: tomar muestras proporcionales a C, para que el pixel se tome bien en la matriz C=1/deltax
deltax_0 = 10E-6                         #paso en espacio
deltax = lamb*z/(N*deltax_0)
R=1E-3
a=0*R

def U0(x, y):               #FUNCIÓN CIRC
    return np.where((np.sqrt((x-a)**2+(y-a)**2) < R) , 0, 1)       


def U01(x, y):              #FUNCIÓN RECT
    return np.where(((np.abs(x)<R) & (np.abs(y)<R)), 1, 0)


#Fase parabolica en el plano Z=0
def Fase2(x_0,y_0):
    return np.exp((1j*k/(2*z))*(x_0**2+y_0**2))

# Fase parabólica en el plano de salida. 
def ConstantPhase(x,y):
    return (np.exp(1j*k*z))*(np.exp((1j*k/(2*z))*(x**2+y**2)))/(1j*lamb*z)

# Producto entre fase parabólica por el la transmitancia en z=0
def U_1(x_0,y_0):
    Product=(deltax**2)*Fase2(x_0,y_0)*U0(x_0,y_0)
    return Product

def main():
    x=np.arange(-N//2,N//2)
    print(len(x))
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