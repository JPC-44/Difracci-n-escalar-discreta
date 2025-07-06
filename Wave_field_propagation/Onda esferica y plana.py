import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros de la onda
A1 = 8
A2= 8     # Amplitud

kx1 = 2 * np.pi / 2.5  # Número de onda en la dirección x (2π/longitud de onda)
ky1 = 2 * np.pi / 2.5 # Número de onda en la dirección y (2π/longitud de onda)

ky2= 2 * np.pi / 10
kx2= np.sqrt((2*np.pi/2.5)**2-ky2**2)

omega1 = 2 * np.pi / 5  # Frecuencia angular (2π/período)
omega2 = 2 * np.pi / 5
phi =np.pi/2   # Fase inicial
E=0
onda1=1
onda2=1
# Configuración del espacio y tiempo
x = np.linspace(0, 20, 300)  # Posiciones de x
y = np.linspace(0, 20, 300)  # Posiciones de y
t = np.linspace(0, 10, 100)  # Intervalo de tiempo

X, Y = np.meshgrid(x, y)


# Función para la onda plana en 2D

#def onda_plana(X, Y, t):
#    return   (onda1*A1*np.sin(kx1 * X + ky1 * Y - omega1 *E* t + phi)+onda2*A2*np.sin(kx2*X+ky2*Y-omega2*E*t))

def onda_plana(X, Y, t):  #ONDA ESFERICA
     return   (onda1*A1*np.sin(np.sqrt((kx1**2+ky1**2)*((X)**2+(Y-10)**2)) - omega1 *E* t + phi)+onda2*A2*np.sin(np.sqrt((kx2**2+ky2**2)*((X)**2+(Y-11)**2))-omega2*E*t)+
               onda2*A2*np.sin(np.sqrt((kx2**2+ky2**2)*((X)**2+(Y-12)**2))-omega2*E*t)+onda2*A2*np.sin(np.sqrt((kx2**2+ky2**2)*((X)**2+(Y-12)**2))-omega2*E*t))

# Configuración de la figura
fig, ax = plt.subplots()
cax = ax.imshow(onda_plana(X, Y, 0), extent=[0, 20, 0, 20], vmin=-10, vmax=10, cmap='viridis')
fig.colorbar(cax)

# Función de actualización para la animación
def actualizar(frame):
    ax.clear()
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    cax = ax.imshow(onda_plana(X, Y, t[frame]), extent=[0, 20, 0, 20], vmin=-10, vmax=10, cmap='viridis')
    return cax,

# Animación
anim = FuncAnimation(fig, actualizar, frames=len(t), interval=60, blit=False)

# Mostrar la animación
plt.show()