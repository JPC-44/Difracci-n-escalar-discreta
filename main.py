import os
import numpy as np
from numerical_diffraction.utils import general_functions as funcs
import cv2
import sys
import matplotlib.pyplot as plt

def main():
    # --------- Parámetros ---------
    λ = 633e-9  # Longitud de onda en metros
    ic_capture4000x3000_pixel_size = 1.85E-6 
    thorlabs_pixel_size = 5.2E-6  # Tamaño de píxel en metros
    
    pixel_size = ic_capture4000x3000_pixel_size
    modo = 'r'
    metodo = 'as'
    numero_carpeta = int(1)

    # modo = input("Elige modo (p=propagación, r=retropropagación): ").strip().lower()
    # retro = True if modo == "r" else False
    # metodo = input("Elige método (AS:AngularSpectrum, FC:FresnelConvolution, FT:FresnelTransformation): ").strip().lower()

    # numero_carpeta = int(funcs.get_int_input("Ingrese la carpeta de trabajo imagenes_entradaXX"))

    print('Parámetros de trabajo')
    print(f'λ = {λ*1E+9} nm')
    print(f'pixel size = {pixel_size*1E+6} um')

    if modo == 'r':
        print('Iniciando la retropropagación')
        retro = True
    elif modo == 'p':
        print('Iniciando la propagación')
        retro = False

    if metodo == 'as':
        print('Iniciando con Angular Spectrum method')
    elif metodo == 'fc':
        print('Iniciando Fresnel Convolution')
    elif metodo == 'ft':
        print('Iniciando Fresnel Transform')
    

    start_z = funcs.get_float_input("Ingrese el valor inicial de Z (en metros): ")
    step_z = funcs.get_float_input("Ingrese el paso entre valores de Z (en metros): ")
    stop_z = funcs.get_float_input("Ingrese el valor final de Z (en metros): ")
    num_images = int((stop_z - start_z) / step_z) + 1    # Calcular número de imágenes a generar
    print(f"Se generarán {num_images} imágenes.")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    carpeta_in = script_dir + f"/results" + f"/imagenes_entrada{numero_carpeta}/campo_a_propagar"
   
    carpeta_out  = script_dir + f"/results" + f"/imagenes_entrada{numero_carpeta}/imagenes_calculadas"
    
    os.makedirs(carpeta_out, exist_ok=True)

    # --------- Carga de imágenes ---------
    real_img = funcs.cargar_imagen(os.path.join(carpeta_in, "real.png"))
    imag_img = funcs.cargar_imagen(os.path.join(carpeta_in, "imag.png"))
    intensity = funcs.cargar_imagen(os.path.join(carpeta_in,"intensity.jpg"))


    create_optical_field = funcs.crear_campo(real_img, imag_img, intensity)


    entrance_optical_field = np.array(create_optical_field)
    
    M, N = entrance_optical_field.shape
    print(f'Tamaño de la imagen: {N*pixel_size*1000, M*pixel_size*1000} mm')

    imagenes = []  # stack de imagenes 
    labels_posiciones = [] # label con posiciones z

    if metodo == 'as':
        
        for i in range(0,num_images):
            z = start_z + i * step_z

            optical_field_at_plane_z = funcs.AngularSpectrum(entrance_optical_field, z, λ, pixel_size, retro)
            intensity = np.abs(optical_field_at_plane_z)**2
            imagenes.append(intensity)
            labels_posiciones.append(f"{z:.5f}")
        funcs.slice_images(imagenes, labels_posiciones)

    elif metodo == 'fc':
        for i in range(0,num_images):
            z = start_z + i * step_z

            optical_field_at_plane_z = funcs.Fresnel_convolution(entrance_optical_field, z, λ, pixel_size, retro)
            intensity = np.abs(optical_field_at_plane_z)**2
            imagenes.append(intensity)
            labels_posiciones.append(f"{z:.5f}")
        funcs.slice_images(imagenes, labels_posiciones)
            
    elif metodo == 'ft':
        for i in range(0,num_images):
            z = start_z + i * step_z

            optical_field_at_plane_z, coords = funcs.Fresnel_transformation(entrance_optical_field, z, λ, pixel_size)
            intensity = np.abs(optical_field_at_plane_z)**2
            imagenes.append(intensity)
            labels_posiciones.append(f"{z:.5f}")
        funcs.slice_images(imagenes, labels_posiciones)

    guardar = input("Ingrese (si) o (no) para guardar el conjunto de imágenes: ").strip().lower()

    if guardar == 'si':
        funcs.guardar_imagenes(imagenes, labels_posiciones, carpeta_out)

    restar_fondo = input("Desea restarle el fondo? (si) (no)").strip().lower()


    if restar_fondo == 'si':
        
        medida_en_metros = str(input("Seleccione la distancia float de .5f de la imagen z_0.XXXXX").strip().lower())
        z_ = float(medida_en_metros)
        recons = cv2.imread(f"{carpeta_out}/z_{medida_en_metros}m.png", cv2.IMREAD_GRAYSCALE)
        fondo = cv2.imread(f"{carpeta_in}/fondo.jpg", cv2.IMREAD_GRAYSCALE)
        print("reconstrucción:",f"{carpeta_in}/z_{medida_en_metros}m.png")
        print("fondo:",f"{carpeta_in}/fondo.jpg")
      

        plt.imshow(recons, cmap='gray')
        plt.show()
        plt.imshow(fondo, cmap='gray')
        plt.show()
        resta = np.array(recons)-np.array(fondo)
        plt.imshow(resta,cmap='gray')
        plt.title("resta de reconstrucción y fondo")
        plt.show()
        # if metodo == 'as':
        #     z = z_
        #     optical_field_at_plane_z = funcs.AngularSpectrum(entrance_optical_field, z, λ, pixel_size, retro)
        #     intensity = np.abs(optical_field_at_plane_z)**2

        #     labels_posiciones.append(f"{z:.5f}")
        # funcs.slice_images(imagenes, labels_posiciones)
         

if __name__ == "__main__":
    
    main()
