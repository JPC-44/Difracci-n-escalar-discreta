import numpy as np
import matplotlib.pyplot as plt





def dft2d_optimized(image):

    M, N = image.shape # tamaño de la imagne a la que se le implementa la DFT

    dft_result = np.zeros((M, N), dtype=complex)          # creación de array en ceros de tipo complejo para la DFT
    
    x=np.arange(N)
    y=np.arange(N)


    Wx = np.exp(-2j * np.pi * np.outer(x, x) / M)        #base  en x 
    Wy = np.exp(-2j * np.pi * np.outer(y, y) / N)         # base en y  

    temp = np.dot(Wx, image)
    
    dft_result = np.dot(temp, Wy)
    
    return ((deltax*N)**2)*dft_result

def shift(arr):                                  #shift de la DFT
    N = len(arr)
    mid = N // 2   #división que toma el valor por debajo // 
    if N % 2 == 0:   #   residuo %
        return np.concatenate((arr[mid:], arr[:mid]))    
    else:
        return np.concatenate((arr[mid+1:], arr[:mid+1]))

def shift2D(A):
    Q=len(A)//2             
    N=len(A)
    S=[]
    for p in range(0,N):
        S.append(shift(A[p]))
    for p in range(N-1,-1,-1):
        SS=shift(S)         #salen las frecuencias desfasadas
    SI=[]                          #shift sin el desfase para que cuadren las frecuencia
    for p in range(0,N):
        B=[]
        for q in range(N-1,-1,-1):
            B.append(SS[p][q])
        SI.append(B)
    return SI


z=10E-2    #distancia entre planos
lamb=650E-9                            #longitud de onda que se propaga
N=1024       #  Muestras 
1
R=0.2E-3 
def U0(x, y):              #YOUNG
    return np.where(((np.abs(x)<R) & (np.abs(y)<R)),1, 0)

deltax=10E-6  
deltay=deltax
x=np.arange(-N//2,N//2)
y=np.arange(-N//2,N//2)
x,y=x*deltax,y*deltax
X,Y=np.meshgrid(x,y)

U=U0(X,Y)


plt.imshow(U,cmap='gray')
plt.show()

FFT=np.fft.fft2(U)

plt.imshow((np.abs(FFT)),cmap='gray')
plt.show()


plt.imshow(np.fft.fftshift(np.abs(FFT)),cmap='gray')
plt.show()

print(np.sum(U**2))
print(np.sum(np.abs(FFT)**2))