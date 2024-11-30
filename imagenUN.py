from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


ruta_imagen = "C:/Users/Julian/Desktop/OD/IMAGENUN.png"
imagen = Image.open(ruta_imagen)
imagen_gris = imagen.convert('L')     #escala grises- libreria pil -------- paso necesario para poder multiplicar matrices abajo A0*H0
umbral = 100       # desde un valor de umbral todos hacia arriba blancos y por debajo, negros
imagen_bn = imagen_gris.point(lambda p: p > umbral and 255)  #aplicacion del umbral
matriz_bn = np.array(imagen_bn)
print(matriz_bn.shape)  #forma (ancho,largo,colores)

z=0.03          #distancia entre planos
lamb=633E-9                            #longitud de onda que se propaga
N=830        #  Muestras      NOTA: tomar muestras proporcionales a C, para que el pixel se tome bien en la matriz C=1/deltax
                     # espacio de las frecuencias va desde [0,C] cuando no está centrada
deltax=4.82E-6                        #paso en espacio
deltay=deltax
deltafx=1/(N*deltax)
deltafy=deltafx             
mid=N//2

#def U0(x, y):               #FUNCIÓN CIRC
#    R=0.63E-3
#    return np.where((np.sqrt((x)**2+(y)**2) < R) , 0, 1)       


#def U0(x, y):              #FUNCIÓN RECT
#    return np.where(((np.abs(x)<0.63E-3) & (np.abs(y)<0.63E-3)), 1, 0)


def shift(arr):                                  #shift de la DFT
        N = len(arr)
        mid = N // 2   #división que toma el valor por debajo // 
        if N % 2 == 0:   #   residuo %
            return np.concatenate((arr[mid:], arr[:mid]))    # toma el indice n=mid y toma valores [mid,....N), [0,...mid) para N entero
        else:
            return np.concatenate((arr[mid+1:], arr[:mid+1]))
def shift2D(A):                                  #shift para centrar freq en matrices
    def shift(arr):                                  #shift de la DFT
        N = len(arr)
        mid = N // 2   #división que toma el valor por debajo // 
        if N % 2 == 0:   #   residuo %
            return np.concatenate((arr[mid:], arr[:mid]))    # toma el indice n=mid y toma valores [mid,....N), [0,...mid) para N entero
        else:
            return np.concatenate((arr[mid+1:], arr[:mid+1]))    
    N=len(A)
    S=[]
    for p in range(0,N):
        S.append(shift(A[p]))
    SS=shift(S)         
    return SS
def linspace(N,d):     #   N= numero de muestras,     d= tamaño de intervalo para x,  d=deltax*N, para f, d=1/deltax  -crear espacio
    x1=[] 
    if N%2==0: 
        for i in range(0,N//2):
            x1.append(i*d/N)
        for j in range(-N//2,0,1):
            x1.append(j*d/N)
    else:
        for i in range(0,N//2+1):
            x1.append(i*d/N)
        for j in range(-N//2+1,0,1):
            x1.append(j*d/N)
    x2=shift(x1)
    return x2
def circ(X,Y):
    distancia = np.sqrt(X**2 + Y**2)
    circulo = np.where(distancia <= 1, 1, 0)
    return circulo
def H(fx,fy):                      #función transferencia del espacio libre
      return (np.exp((1j*2*np.pi*z/lamb)*np.sqrt(1-((lamb)**2)*(fx**2+fy**2))))*circ(lamb*fx,lamb*fy)

x=linspace(N,deltax*N)
y=linspace(N,deltax*N)             #linspace en el espacio
X,Y=np.meshgrid(x,y)

fx=linspace(N,1/deltax)
fy=linspace(N,1/deltax)        #LINSPACE EN ESPACIO DE FRECUENCIAS
Fx,Fy=np.meshgrid(fx,fy)

Z0=matriz_bn                                   #tener en cuenta ifft recibe matrices de espacio descentrada
H0=H(Fx,Fy)                               #TENER EN CUENTA QUE LA fft recibe matrices con frecuencia descentrada
plt.imshow(np.angle(H0),cmap='gray')
plt.show()
A0=shift2D(np.fft.fft2(Z0))  
      
#para realizar el producto de A0*H0 "filtro" se debe centrar A0 para que las frecuencias coincidan con H0
# luego se vuelve a descentrar las frecuencias para ingresarlas a la ifft2D 
U=(np.fft.ifft2((shift2D(A0*H0))))        #U se devuelve centrada


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(Z0**2, extent=(min(x), max(x), min(y), max(y)), origin='upper', cmap='gray')
plt.colorbar()
plt.title('Campo en z=0 -logo UN')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.imshow((np.abs(U))**2, extent=(min(x), max(x), min(x), max(x)), origin='upper', cmap='gray')
plt.colorbar()
plt.title('Campo propagado z=z0')
plt.xlabel('fx')
plt.ylabel('fy')

plt.tight_layout()
plt.show()


