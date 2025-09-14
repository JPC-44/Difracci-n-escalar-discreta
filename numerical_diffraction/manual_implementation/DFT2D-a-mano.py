import cmath
import numpy as np
import matplotlib.pyplot as plt





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

def base(X,Y,Z,T,N):                                    #base posición X=x, frecuencia deltafx=Y
   return np.exp(-2*np.pi*1j*(X*Y+Z*T)/N)     
                                                  # para obtener la parte real: Function.real
                                                  #de qué función quiero obtener la DFT?

#def f(x, y):                                 #funcion circ
#    return np.where(x**2 + y**2 < 1, 1, 0)

def f(x, y):         # funcion rect

    return np.where((np.abs(x) < 1) & (np.abs(y) < 1), 1, 0)

#def f(x, y):         # func cosenos

#    return np.cos(2*np.pi*10*x)+np.cos(2*np.pi*15*y)


N=30                    #  Muestras=datos  
C=30             #           #ancho de espacio frecuencial (0,C)=(0,N*deltafx)   de donde a donde van las frecuencias C=N*deltafx  
deltax=1/C                         #paso en espacio
deltay=1/C
deltafx=C/N                    # paso en espacio frecuencias
deltafy=C/N

def A0(p,q):                  #ESPECTRO ANGULAR EN z=0
   suma=0
   for m in range(0,N):
        for n in range(0,N):
            suma+=f(deltax*n,deltay*m) * base(n,p,m,q,N)
   return suma



DFT2d=[]                           #DFT sin centrar
for p in range(0,N):
   DFTp=[]
   for q in range(0,N):
      DFTp.append(A0(p,q))
   DFT2d.append(DFTp)

DFT2dshifted=shift2D(DFT2d)              #DFT centrada    parte imaginaria+real

DFT2dabs=[]                 # parte abs de la DFT           


for p in range(0,N):
   DFTp=[]
   for q in range(0,N):
      DFTp.append(abs((DFT2dshifted[p][q])))
   DFT2dabs.append(DFTp) 



fx_values=[]
for i in range(0,N):
    fx_values.append(deltafx*(i-N//2+1))
fy_values=[]
for i in range(0,N):
    fy_values.append(deltafy*(i-N//2+1))
print(DFT2dabs)
DFT2dabs=np.abs(DFT2dabs)

# Graficar la matriz 2D usando imshow con coordenadas especificadas

""" plt.imshow(np.log2(DFT2dabs+1), cmap='gray')#, aspect='auto', extent=[min(fx_values), max(fx_values), min(fy_values), max(fy_values)])
plt.colorbar(label='Valor')
plt.title('DFT parte real')
plt.xlabel('Coordenada fx')
plt.ylabel('Coordenada fy')
plt.show() """
                












