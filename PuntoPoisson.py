import numpy as np
import matplotlib.pyplot as plt
import sys

z=0.15         #distancia entre planos
lamb=632E-9                            #longitud de onda que se propaga
N=512      #  Muestras      NOTA: tomar muestras proporcionales a C, para que el pixel se tome bien en la matriz C=1/deltax
#C=100                      # espacio de las frecuencias va desde [0,C] cuando no está centrada
deltax=10E-6                    #unidades 10um
deltay=deltax
deltafx=1/(N*deltax)                 #tamaño del pixel 200[1/m]
deltafy=deltafx             
mid=N//2
R=0.6E-3               # 6 mm

if R < deltax*(N//2):
    print('delta*(N//2) =',deltax*N)
    print('Radio =', R)
else: 
    sys.exit()  # Sale del programa





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



def U0(x, y):               #funcion a propagar
    return np.where(((x)**2+(y)**2 < R**2) , 0, 1)  
  
def U01(x, y):              #FUNCIÓN RECT
    return np.where(((np.abs(x)<R) & (np.abs(y)<R)), 1, 0)   

def circ(X,Y):          # función para filtrar 
    distancia = np.sqrt((X)**2 + (Y)**2)
    circulo = np.where(distancia <= 1, 1, 0)
    return circulo

def H(fx,fy):                      #función transferencia del espacio libre
      return np.where(1-((lamb)**2)*(fx**2+fy**2)>0,(np.exp((1j*2*np.pi*z/lamb)*np.sqrt(1-((lamb)**2)*(fx**2+fy**2)))),0)   #*circ(lamb*fx,lamb*fy)


x=np.arange(-N//2,N//2+1)
y=np.arange(-N//2,N//2+1)
x,y=x*deltax,y*deltax
X,Y=np.meshgrid(x,y)

fx=np.arange(-N//2,N//2+1)
fy=np.arange(-N//2,N//2+1)        #LINSPACE EN ESPACIO DE FRECUENCIAS
fx,fy=fx*deltafx,fy*deltafx
Fx,Fy=np.meshgrid(fx,fy)



Z0=U0(X,Y)                                    #tener en cuenta ifft recibe matrices de espacio descentrada
H0=H(Fx,Fy)                                 #TENER EN CUENTA QUE LA fft recibe matrices con frecuencia descentrada
A0=np.fft.fftshift(np.fft.fft2(Z0))                   
#para realizar el producto de A0*H0 "filtro" se debe centrar A0 para que las frecuencias coincidan con H0
# luego se vuelve a descentrar las frecuencias para ingresarlas a la ifft2D 
U=(np.fft.ifft2((np.fft.fftshift(A0*H0))))        #U se devuelve centrada

#plt.imshow(np.log(np.abs(A0)**2+1),cmap='gray')
#plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(Z0**2, extent=(min(x), max(x), min(y), max(y)), origin='upper', cmap='gray')
plt.colorbar()
plt.title('Campo en z=0 -Obstaculo en el medio')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.imshow((np.abs(U)**2),extent=(min(x), max(x), min(y), max(y)), origin='upper', cmap='gray') #extent=(min(fx), max(fx), min(fx), max(fx))
plt.colorbar()
plt.title('Campo propagado z=z0  -Punto Poisson')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()