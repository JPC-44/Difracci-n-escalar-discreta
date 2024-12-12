import numpy as np
import matplotlib.pyplot as plt


z=10E-2                 # Distancia entre planos
lamb=650E-9             # Longitud de onda que se propaga
N=1024                  # Muestras 
deltax=10E-6            # Paso en x
deltay=deltax           # Paso en y
deltafx=1/(N*deltax)    # Paso fx
deltafy=deltafx         # Paso fy
m=1            
L=2E-3


    
def U0(x, y):              
    return 0.5*(1+m*np.cos(2*np.pi*x/L+0*y))



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

    # Función máscara evaluada en el meshgrid
    Z0=U0(X,Y)  
    #Funcion de transferencia evaluada en el meshgrid
    H0=H(Fx,Fy)
    
    A0_unshifted=np.fft.fft2(Z0)            # Espectro angular en z=0 descentrado
    A0=np.fft.fftshift(A0_unshifted)        # Se centra el espectro angular 
    A_shifted=np.multiply(A0,H0)            # Se obtiene el espectro angular en z 
    A_unshifted=np.fft.fftshift(A_shifted)  # Se descentran las freq. para usar ift
    U=np.fft.ifft2(A_unshifted)             # Se calcula el campo propagado U a una distancia z

    if z<N*deltax**2/lamb:                              # condición donde empieza a fallar espectro angular
        print(f'z es mayor que {N*deltax**2/lamb}')
    Espectro=((np.abs(U))**2)                           # Se calcula el espectro o modulo cuadrado
    
    # SE REALIZAN LOS SIGUIENTES PLOTs. con titulos descriptivos

    plt.imshow(np.angle(H0),cmap='gray')
    plt.title('Fase de función de transferencia')
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(Z0, extent=(min(x), max(x), min(y), max(y)), origin='upper', cmap='gray')
    plt.colorbar()
    plt.title('Módulo cuadrado en z=0')
    plt.xlabel('x0[m]')
    plt.ylabel('y0[m]')

    plt.subplot(1, 2, 2)
    plt.imshow(Espectro, extent=(min(x), max(x), min(x), max(x)),origin='upper', cmap='gray',vmax=0.05)
    plt.colorbar()
    plt.title(f'Módulo cuadrado de campo propagado z0={z} metros.')
    plt.xlabel('x[m]')
    plt.ylabel('y[m]')

    plt.tight_layout()
    plt.show()

    # VERIFICACIÓN DE ENERGÍA EN Z=0 Y EN Z=z
    print(np.sum(Espectro))
    print(np.sum(Z0**2))

# Se llama la función que ejecuta el proceso de propagación numérica    
main()