import numpy as np
import matplotlib.pyplot as plt
z=20E-2    #distancia entre planos
lamb=650E-9                            #longitud de onda que se propaga
N=2048       #  Muestras      NOTA: tomar muestras proporcionales a C, para que el pixel se tome bien en la matriz C=1/deltax
#C=100                      # espacio de las frecuencias va desde [0,C] cuando no está centrada
deltax=100000E-6                         #paso en espacio
deltay=deltax           #paso en y
deltafx=1/(N*deltax)    #paso frecuencial en x
deltafy=deltafx         #paso frecuencial en y
R=0.5E-3        #Radio de la apertura circular
a=0*R  

lamb=0.1
def U0(x, y):              #rect
    return np.where((1/lamb**2-x**2-y**2>0),np.cos((2/lamb)*np.pi*(x+y))*np.cos(2*np.pi*(np.sqrt(1/lamb**2-x**2-y**2))/lamb)-np.sin(2*np.pi*(x+y)/lamb)*np.sin((2)*np.pi*(np.sqrt(1/lamb**2-x**2-y**2))), 0) 

x=np.arange(-N//2,N//2)
y=np.arange(-N//2,N//2)
x,y=x*deltax,y*deltax
X,Y=np.meshgrid(x,y)

U=U0(X,Y)

plt.imshow(np.real(U),cmap='gray',vmax=0.1)
plt.show()