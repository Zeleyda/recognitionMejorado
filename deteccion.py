import os
import cv2
import face_recognition
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Cargar la imagen a comparar
input_image_path = input("Imagen a comparar: ")
input_image = load_image(input_image_path)
input_face_locations = face_recognition.face_locations(input_image)

resultados = []

if len(input_face_locations) == 0:
    print("No se ha encontrado ningún rostro en la imagen a comparar.")
else:
    input_face_encoding = face_recognition.face_encodings(input_image, input_face_locations)[0]

    # Iterar a través de las subcarpetas en la carpeta principal
    folder_path = "ruta"
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Asegúrate de leer solo archivos de imagen
                image_path = os.path.join(root, filename)
                image = load_image(image_path)
                face_locations = face_recognition.face_locations(image)

                if len(face_locations) == 0:
                    print(f"No se ha encontrado ningún rostro en {filename}.")
                else:
                    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                    match = face_recognition.compare_faces([input_face_encoding], face_encoding, tolerance=0.6)

                    if match[0]:
                        resultados.append(image_path)

if(len(resultados) > 0):
    print('La imagen proporcionada a coincidido con las siguientes imagenes:\n')
    print(resultados)
    print('\n\n')
else:
    print('No se encontraron coincidencias :(')
