import numpy as np
import matplotlib.pyplot as plt


z=0.1     #distancia entre planos
lamb=500E-9                            #longitud de onda que se propaga
k=2*np.pi/lamb
N=1024       #  Muestras      NOTA: tomar muestras proporcionales a C, para que el pixel se tome bien en la matriz C=1/deltax
#C=100                      # espacio de las frecuencias va desde [0,C] cuando no está centrada
deltax_0 = 8E-6                         #paso en espacio
deltax = lamb*z/(N*deltax_0)
R=1E-3
a=1*R


def U0(x, y):               #FUNCIÓN CIRC
    return np.where((np.sqrt((x-a)**2+(y-a)**2) < R) , 1, 0)       


def U01(x, y):              #FUNCIÓN RECT
    return np.where(((np.abs(x)<R) & (np.abs(y)<R)), 1, 0)


#FASE CUADRÁTICA

def Fase2(x_0,y_0):
    return np.exp((1j*k/(2*z))*(x_0**2+y_0**2))



def ConstantPhase(x,y):
    return (np.exp(1j*k*z))*(np.exp((1j*k/(2*z))*(x**2+y**2)))/(1j*lamb*z)

def U_1(x_0,y_0):
    Product=(deltax**2)*Fase2(x_0,y_0)*U0(x_0,y_0)
    return Product

def main():
    x=np.arange(-N//2,N//2+1)
    y=np.arange(-N//2,N//2+1)
    x,y=x*deltax,y*deltax
    X,Y=np.meshgrid(x,y)

    x_0=np.arange(-N//2,N//2+1)
    y_0=np.arange(-N//2,N//2+1)       
    x_0,y_0=x_0*deltax_0,y_0*deltax_0
    X_0,Y_0=np.meshgrid(x_0,y_0)

    U1=U_1(X_0,Y_0)       

    F0=np.fft.fftshift(np.fft.fft2(U1))
    F1=np.multiply(F0,ConstantPhase(X,Y))                
    #para realizar el producto de A0*H0 "filtro" se debe centrar A0 para que las frecuencias coincidan con H0
    # luego se vuelve a descentrar las frecuencias para ingresarlas a la ifft2D 
    #U se devuelve centrada


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(U1)**2, extent=(min(x), max(x), min(y), max(y)), origin='upper', cmap='gray')
    plt.colorbar()
    plt.title('')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)
    plt.imshow((((np.abs(F1))**2)), extent=(min(x), max(x), min(x), max(x)), origin='upper', cmap='gray')
    plt.colorbar()
    plt.title(f'Campo propagado z0={z} metros.')
    plt.xlabel('fx')
    plt.ylabel('fy')

    plt.tight_layout()
    plt.show()
    return 0

if z<N*deltax**2/lamb:
    main()
else:
    print(f'z es mayor que {N*deltax**2/lamb}')