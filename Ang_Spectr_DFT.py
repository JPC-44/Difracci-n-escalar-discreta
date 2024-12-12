import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

image = Image.open('Logo_OD.png').convert('L')  # Convertir a escala de grises
Z0 = np.sqrt(np.array(image, dtype=np.float64))


z=0.15                 # Distancia entre planos
lamb=650E-9             # Longitud de onda que se propaga
M, N = Z0.shape
deltax=10E-6            # Paso en x
deltay=deltax           # Paso en y
deltafx=1/(N*deltax)    # Paso fx
deltafy=deltafx         # Paso fy




# Función DFT bidimensional
def dft2d(image):
    # Creación de array en ceros de tipo complejo para la DFT.
    dft2D = np.zeros((M, N), dtype=complex)         
    
    x1 = np.arange(M)       # Listas del 0,...,1023
    y1 = np.arange(N)        

    # Matrices con la base o kernel de Fourier
    Kernel_x = np.exp(-2j * np.pi * np.outer(x1, x1) / M) #base  en x 
    Kernel_y = np.exp(-2j * np.pi * np.outer(y1, y1) / N) # base en y  
    
    # aplicación de transformada unidimensional
    dft_x = np.dot(Kernel_x, image)
    # aplicación de transfomada bidimensional
    dft2D = np.dot(dft_x, Kernel_y)

    return dft2D

# Función que realiza transformada de Fourier inversa
def inverse_dft2d(image):    # el input debe estar unshifted


    dft2D = np.zeros((M, N), dtype=complex)         
    
    x1 = np.arange(M)       # Listas del 0,...,1023
    y1 = np.arange(N)        

    # Matrices con la base o kernel conjugado de Fourier
    Kernel_x = np.exp(2j * np.pi * np.outer(x1, x1) / M) #base  en x 
    Kernel_y = np.exp(2j * np.pi * np.outer(y1, y1) / N) # base en y  
    
    # aplicación de transformada unidimensional
    dft_x = np.dot(Kernel_x, image)
    # aplicación de transfomada bidimensional
    dft2D = np.dot(dft_x, Kernel_y)
    
    return deltafx**2*deltax**2*dft2D

#Función shift de la DFT Unidimensional
def shift(arr):      
    N = len(arr)
    mid = N // 2   #división que toma el valor por debajo // 
    if N % 2 == 0:   #   residuo %
        return np.concatenate((arr[mid:], arr[:mid]))    
    else:
        return np.concatenate((arr[mid+1:], arr[:mid+1]))

#Función shift bidimensional de la DFT2D
def shift2D(A):
           
    N=len(A)
    S=[]
    for p in range(0,N):
        S.append(shift(A[p]))
    for p in range(N-1,-1,-1):
        SS=shift(S)                #salen las frecuencias desfasadas
    SI=[]                          #shift sin el desfase para que cuadren las frecuencia
    for p in range(0,N):
        B=[]
        for q in range(N-1,-1,-1):
            B.append(SS[p][q])
        SI.append(B)
    return SI


# Función circ para filtrado de las frecuencias no propagantes de H(fx,fy)
def circ(X,Y):
    distancia = np.sqrt(X**2 + Y**2)
    circulo = np.where(distancia <= 1, 1, 0)
    return circulo

# Función transferencia del espacio libre
def H(fx,fy):              
      return (np.exp((1j*2*np.pi*z/lamb)*np.sqrt(1-((lamb)**2)*(fx**2+fy**2))))*circ(lamb*fx,lamb*fy)


# Función main donde se ejecuta en meshgrid y las operaciones para propagar espectro angular
def main():
    # Marca de tiempo inicial
    start_time = time.perf_counter()
    print("Inicio del proceso...")


    #LINSPACE EN ESPACIO 
    x=np.arange(-N//2,N//2)
    y=np.arange(-N//2,N//2)
    x,y=x*deltax,y*deltax
    X,Y=np.meshgrid(x,y)

    #LINSPACE EN ESPACIO DE FRECUENCIAS
    fx=np.arange(-N//2,N//2)
    fy=np.arange(-N//2,N//2)        
    fx,fy=fx*deltafx,fy*deltafx
    Fx,Fy=np.meshgrid(fx,fy)


    #Funcion de transferencia evaluada en el meshgrid
    H0=H(Fx,Fy)
    A0_unshifted=dft2d(Z0)                          # Espectro angular en z=0 descentrado
    A0=np.fft.fftshift(A0_unshifted)                # Se centra el espectro angular 
    A_shifted=np.multiply(A0,H0)                    # Se obtiene el espectro angular en z 
    A_unshifted=np.fft.fftshift(A_shifted)          # Se descentran las freq. para usar ift
    U=inverse_dft2d(np.fft.fftshift(A_unshifted))   # Se calcula el campo propagado U a una distancia z

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Tiempo transcurrido espectro angular con DFT: {elapsed_time:.8f} segundos")


    if z<N*deltax**2/lamb:                          # condición donde empieza a fallar espectro angular
        print(f'z es mayor que {N*deltax**2/lamb}')
    
    Espectro=((np.abs(U))**2)                       # Se calcula el espectro o modulo cuadrado
    
    # SE REALIZAN LOS SIGUIENTES PLOTs. con titulos descriptivos
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(Z0, extent=(min(x), max(x), min(y), max(y)), origin='upper', cmap='gray')
    plt.colorbar()
    plt.title('')
    plt.xlabel('x0')
    plt.ylabel('y0')

    plt.subplot(1, 2, 2)
    plt.imshow(Espectro, extent=(min(x), max(x), min(x), max(x)),origin='upper', cmap='gray')
    plt.colorbar()
    plt.title(f'Campo propagado z={z} metros.')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.show()

    # VERIFICACIÓN DE ENERGÍA EN Z=0 Y EN Z=z
    #print(np.sum(Espectro))
    #print(np.sum(Z0**2))

# Se llama la función que ejecuta el proceso de propagación numérica    
main()