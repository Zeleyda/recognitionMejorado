import numpy as np
import face_recognition
import os

def get_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    return face_encodings

def compare_face_encodings(encoding1, encoding2_path, tolerance=0.5):
    encoding2 = np.load(encoding2_path)
    result = face_recognition.compare_faces(encoding1, encoding2, tolerance=tolerance)
    return result[0]

if __name__ == "__main__":
    image_path = input("Ruta de la imagen: ")
    input_image_enconding = get_face_encoding(image_path)

    results = []
    if  len (input_image_enconding) > 0:
        folder = os.listdir('encondings2')
        for file in folder:
            if file == 'tmp.npy':
                continue
            if compare_face_encodings(input_image_enconding, 'encondings2/'+file):
                results.append(os.path.splitext(file)[0])
    
    if len (results) == 0:
        print('No se encontraron coincidencias')
    else:
        print('Rostro detectado en las siguientes imagenes: ')
        print(results)