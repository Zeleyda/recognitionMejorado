import cv2
import face_recognition
import os
import numpy as np

def get_face_encoding(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image_rgb)

    if len(face_locations) == 0:
        print(f"No se ha encontrado ningún rostro en {image_path}.")
        return None
    else:
        return face_recognition.face_encodings(image_rgb, face_locations)[0]

# Carga la imagen y obtén el rostro codificado
input_image_path = input("Ruta a la imagen: ")
input_face_encoding = get_face_encoding(input_image_path)

if input_face_encoding is not None:
    # Guarda el rostro codificado en un archivo .npy
    filename = nombre_archivo_sin_extension = os.path.splitext(os.path.basename(input_image_path))[0]

    np.save("encondings/"+filename, input_face_encoding)
    print("Rostro codificado guardado en el archivo: /encondings/"+filename+".npy")

print('\n')
