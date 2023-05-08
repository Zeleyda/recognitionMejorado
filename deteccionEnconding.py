import cv2
import face_recognition
import os
import numpy as np
import glob

def get_face_encoding(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image_rgb)

    if len(face_locations) == 0:
        print(f"No se ha encontrado ningún rostro en {image_path}.")
        return None
    else:
        return face_recognition.face_encodings(image_rgb, face_locations)[0]

def cargar_encodings(ruta_directorio):
    archivos_npy = glob.glob(os.path.join(ruta_directorio, '*.npy'))
    encodings = []
    for archivo in archivos_npy:
        encoding = np.load(archivo)
        encodings.append((archivo, encoding))
    return encodings

def comparar_rostro(face_encoding, encodings, tolerancia=0.6):
    archivos_con_rostro = []
    
    for archivo, encoding in encodings:
        resultado = face_recognition.compare_faces([encoding], face_encoding, tolerance=tolerancia)
        if resultado[0]:
            archivos_con_rostro.append(archivo)
    
    return archivos_con_rostro

def main():
    ruta_directorio_encodings = '/encodings'
    ruta_imagen = input('Ruta de la imagen: ')

    face_encoding = get_face_encoding(ruta_imagen)
    
    if face_encoding is not None:
        encodings = cargar_encodings(ruta_directorio_encodings)
        archivos_con_rostro = comparar_rostro(face_encoding, encodings)
        
        if archivos_con_rostro:
            print("El rostro aparece en los siguientes archivos:")
            for archivo in archivos_con_rostro:
                print(archivo)
        else:
            print("El rostro no aparece en ningún archivo de encoding.")

if __name__ == "__main__":
    main()
