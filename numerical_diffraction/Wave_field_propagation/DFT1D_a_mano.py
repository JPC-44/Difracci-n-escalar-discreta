import cmath
import numpy as np
import matplotlib.pyplot as plt


def base(X,Y):                                    #base posición X=x, frecuencia deltafx=Y
   return np.exp(-2*np.pi*1j*(X*Y))     
                                                
                                                 

def shift(A):                                  #shift de la DFT
    N = len(A)
    mid = N // 2   #división que toma el valor por debajo // 
    if N % 2 == 0:   #   residuo %
        return np.concatenate((A[mid:], A[:mid]))    
    else:
        return np.concatenate((A[mid+1:], A[:mid+1]))

def f(E):                                     # función a la cual se le quiere sacar el contenido frecuencial
   if E>-1 and E<1:
      return 1
   else:
      return 0
   
#def f(E):                                    #  función a la cual se le quiere sacar el contenido frecuencial
#   return np.sin(2*np.pi*100*E)

N=100                     #  Muestras=datos  
C=10             #           #ancho de espacio frecuencial (0,C)=(0,N*deltafx)   de donde a donde van las frecuencias C=N*deltafx  
deltax=1/C                         #paso en espacio
deltafx=C/N                    # paso en espacio frecuencias

def A0(k):                  #ESPECTRO ANGULAR EN z=0
   suma=0
   for n in range(0,N):
      suma+=f(deltax*n)*base(n/N,k)
   return suma
DFT=[]
for k in range(0,N):
   DFT.append(A0(k))

                                    #NOTA: SI deltax ES SUFICIENTEMENTE PEQUEÑO, PUEDO OBTENER MAYOR RESOLUCIÓN QUE SE TRADUCE EN MAYOR ESPACIO FRECUENCIAL
freq=[]         #lista de indices de 0,1,2,3...N-1
for r in range(0,N):
   freq.append(r*deltafx)


freqshift=[]         #lista de indices shifteados, desde -N/2 hasta N/2 
DFTshift=shift(DFT)

for k in range(0,N):      #lista de frecuencias 
   freqshift.append((freq[k]-(N//2)*deltafx))

print(len(freq),len(freqshift),len(DFT),len(DFTshift))


plt.subplot(1,2,1)
plt.plot(freqshift,abs(DFTshift))
plt.xlabel('Frequency')
plt.ylabel('|DFT| centrada')

DFTns=[]    #para graficar la abs(DFT) no centrada
for i in range(0,N):
   DFTns.append(abs(DFT[i]))


plt.subplot(2,2,2)
plt.plot(freq,DFTns)
plt.xlabel('Frequency')
plt.ylabel('|DFT| no centrada')

plt.subplot(2,2,3)
plt.plot(freqshift,np.real(DFTshift))
plt.xlabel('Frequency')
plt.ylabel('real(DFT) centrada')

plt.subplot(2,2,4)
plt.plot(freqshift,np.imag(DFTshift))
plt.xlabel('Frequency')
plt.ylabel('im(DFT) centrada')
plt.show()


