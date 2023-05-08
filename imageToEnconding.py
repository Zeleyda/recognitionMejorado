import face_recognition
import numpy as np
import os

def save_face_encoding(image_path, output_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if len(face_encodings) == 0:
        print(f"No se ha encontrado ningún rostro en {image_path}.")
    else:
        np.save(output_path, face_encodings[0])
        print(f"Características del rostro guardadas en: {output_path}")

if __name__ == "__main__":
    image_path = input("Ruta a la imagen: ")
    filename = nombre_archivo_sin_extension = os.path.splitext(os.path.basename(image_path))[0]
    save_face_encoding(image_path, "encondings2/"+filename+".npy")
