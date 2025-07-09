import numpy as np
import matplotlib.pyplot as plt

def shift2D(A):
    def shift(arr):                                  #shift de la DFT
        N = len(arr)
        mid = N // 2   #división que toma el valor por debajo // 
        if N % 2 == 0:   #   residuo %
            return np.concatenate((arr[mid:], arr[:mid]))    
        else:
            return np.concatenate((arr[mid+1:], arr[:mid+1]))    
    N=len(A)
    S=[]
    for p in range(0,N):
        S.append(shift(A[p]))
    SS=shift(S)         
    return SS

A=[]
N=10
for p in range(0,N):
    B=[]
    for q in range(0,N):
        B.append(f"{p}{q}")
    A.append(B)

Asf=np.fft.fftshift(A)
As=shift2D(A)
print(As)
print(Asf)

A1=[]
for p in range (0,N):
    A2=[]
    for q in range(0,N):
        d=float(Asf[p][q])-float(As[p][q])
        A2.append(d)
    A1.append(A2)
print(A1)