import numpy as np
import matplotlib.pyplot as plt


z=20E-2    #distancia entre planos
lamb=650E-9                            #longitud de onda que se propaga
N=1024       #  Muestras      NOTA: tomar muestras proporcionales a C, para que el pixel se tome bien en la matriz C=1/deltax
#C=100                      # espacio de las frecuencias va desde [0,C] cuando no está centrada
deltax=10E-6                         #paso en espacio
deltay=deltax           #paso en y
deltafx=1/(N*deltax)    #paso frecuencial en x
deltafy=deltafx         #paso frecuencial en y
R=0.5E-3        #Radio de la apertura circular
a=0*R           #SI
def U01(x, y):               #FUNCIÓN CIRC Mascara
    return np.where((np.sqrt((x-a)**2+(y-a)**2) < R) , 1, 0)       

def U02(x, y):              #YOUNG
    return np.where(((np.abs(x)<R) & (np.abs(0.001*y)<R)),1, 0)

def U0(x, y):              #rect
    return np.where(((np.abs(x)<R) & (np.abs(y)<R)),1, 0) 


def circ(X,Y):
    distancia = np.sqrt(X**2 + Y**2)
    circulo = np.where(distancia <= 1, 1, 0)
    return circulo

def H(fx,fy):                      #función transferencia del espacio libre
      return (np.exp((1j*2*np.pi*z/lamb)*np.sqrt(1-((lamb)**2)*(fx**2+fy**2))))*circ(lamb*fx,lamb*fy)

def main():
    x=np.arange(-N//2,N//2)
    y=np.arange(-N//2,N//2)
    x,y=x*deltax,y*deltax
    X,Y=np.meshgrid(x,y)

    fx=np.arange(-N//2,N//2)
    fy=np.arange(-N//2,N//2)        #LINSPACE EN ESPACIO DE FRECUENCIAS
    fx,fy=fx*deltafx,fy*deltafx
    Fx,Fy=np.meshgrid(fx,fy)

    Z0=U0(X,Y)  #Masacara de un ciruculo         #tener en cuenta ifft recibe matrices de espacio descentrada

    H0=H(Fx,Fy)  #Funcion de transferencia      #TENER EN CUENTA QUE LA fft recibe matrices con frecuencia descentrada
    A0=np.fft.fftshift(np.fft.fft2(Z0))     #Calculamos la transformada de fourier de Ao      

    A=np.multiply(A0,H0)

    U=np.fft.ifft2(np.fft.fftshift(A))#*(deltafx**2)        #U se devuelve centrada
    plt.imshow(np.angle(U),cmap='gray')
    plt.show()
    if z<N*deltax**2/lamb:
        print(f'z es mayor que {N*deltax**2/lamb}')
    Espectro=((np.abs(U))**2)
    plt.imshow(np.angle(H0),cmap='gray')
    plt.title('FASE de función de transferencia')
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(Z0, extent=(min(x), max(x), min(y), max(y)), origin='upper', cmap='gray')
    plt.colorbar()
    plt.title('')
    plt.xlabel('x0')
    plt.ylabel('y0')

    plt.subplot(1, 2, 2)
    plt.imshow(Espectro, extent=(min(x), max(x), min(x), max(x)),origin='upper', cmap='gray',vmax=0.05)
    plt.colorbar()
    plt.title(f'Campo propagado z0={z} metros.')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.show()
    print(np.sum(Espectro))
    print(np.sum(Z0**2))
main()