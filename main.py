import os
import numpy as np
import matplotlib.pyplot as plt
from numerical_diffraction.utils import general_functions as funcs


def main():
    # --------- Parámetros ---------
    λ = 633e-9  # Longitud de onda en metros
    pixel_size = 3.45E-6  # Tamaño de píxel en metros

    modo = input("Elige modo (p=propagación, r=retropropagación): ").strip().lower()
    retro = True if modo == "r" else False
    metodo = input("Elige método (AS:AngularSpectrum, FC:FresnelConvolution, FT:FresnelTransformation): ").strip().lower()

    numero_carpeta = int(funcs.get_int_input("Ingrese la carpeta de trabajo imagenes_entradaXX"))
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
    intensity = funcs.cargar_imagen(os.path.join(carpeta_in,"intensity.png"))

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

if __name__ == "__main__":
    main()