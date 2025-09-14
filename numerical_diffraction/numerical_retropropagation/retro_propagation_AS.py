import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys, os

# import de las las funciones generales
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from numerical_diffraction.utils import general_functions as funcs 

def retro_propagation_AS(): 
    # Solicitar la referencia y el número de iteración
    numero_de_iteracion  = int(funcs.get_float_input("Ingrese la carpeta del numero de iteración: "))


    # Crear carpeta para guardar las imágenes
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = script_dir + f"/retropropagacion/iteracion_{numero_de_iteracion}/retro-propagated_images"
    os.makedirs(output_folder, exist_ok=True)




    # variables globales para creación de campo óptico para retropropagación
    real_array = None
    imaginary_array = None
    optical_wave_field  = real_array + 1j * imaginary_array


    try:
        real_path = os.path.join(script_dir, 
                                f"retropropagation/iteration_{numero_de_iteracion}/wave_field_to_retropropagate/real_part.png")


        imaginary_path = os.path.join(script_dir, 
                                f"retropropagation/iteration_{numero_de_iteracion}/wave_field_to_retropropagate/imaginary_part.png")

        real_part = Image.open(real_path).convert('L')  # Convertir a escala de grises
        imaginary_part = Image.open(imaginary_path).convert('L')  # Convertir a escala de grises
        
        real_array = np.array(real_part, dtype=np.float64)
        imaginary_array = np.array(imaginary_part, dtype=np.float64)
        optical_wave_field = real_array + 1j * imaginary_array
        print("Imágenes real e imaginaria cargadas correctamente.")
    except FileNotFoundError:
        raise FileNotFoundError("No se encontró la parte imaginaria, intentando cargar solo la parte real.")


    try:
        real_path = os.path.join(script_dir, 
                            f"retropropagation/iteration_{numero_de_iteracion}/wave_field_to_retropropagate/image_real.png")

        # Cargar solo la parte real si no se encuentran ambas imágenes
        real_part = Image.open(real_path).convert('L')
        real_array = np.array(real_part, dtype=np.float64)

        # matriz de ceros para la parte imaginaria
        imaginary_array = np.zeros_like(real_array, dtype=np.float64)    

        # crear el campo óptico con la parte imaginaria como ceros
        optical_wave_field = real_array + 1j * imaginary_array

        print("No se encontraró la parte imaginaria. Se utilizará solo la parte real.")
    except FileNotFoundError:
        print("No se encontró la parte real tampoco. Asegúrese de que las imágenes existan.")
        sys.exit(0) 


    λ = 633e-9  # Longitud de onda en metros
    pixel_size = 5.2E-6  # Tamaño de píxel en metros

    entrance_optical_field = np.array(np.uint8(optical_wave_field), dtype=np.complex128)  # Usar tipo complejo

    # Parámetros ajustables
    start_z = funcs.get_float_input("Ingrese el valor inicial de retropropagación de Z (en metros): ")
    step_z = funcs.get_float_input("Ingrese el paso entre valores de Z (en metros): ")
    stop_z = funcs.get_float_input("Ingrese el valor final de retropropagación de Z (en metros): ")

    num_images = int((stop_z - start_z) / step_z) + 1    # Calcular número de imágenes a generar
    print(f"Se generarán {num_images} imágenes.")




    # Generar imágenes y guardarlas
    for i in range(num_images):
        z = start_z + i * step_z
        optical_field_at_plane_z = funcs.AngularSpectrum(entrance_optical_field, z, λ, pixel_size)
        intensity = np.abs(optical_field_at_plane_z)**2

        output_path = os.path.join(output_folder, f"retropropagated_Z_{z:.5f}m.png")
        plt.imsave(output_path, intensity, cmap='gray')
        print(f"Imagen de intensidad guardada: {output_path}")

    print(f"Imágenes generadas y guardadas en la carpeta: {output_folder}")
