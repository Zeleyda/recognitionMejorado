import numpy as np
import face_recognition

def compare_face_encodings(encoding1_path, encoding2_path, tolerance=0.6):
    encoding1 = np.load(encoding1_path)
    encoding2 = np.load(encoding2_path)
    result = face_recognition.compare_faces([encoding1], encoding2, tolerance=tolerance)
    return result[0]

if __name__ == "__main__":
    encoding1_path = input("Ruta del archivo .npy de encoding 1: ")
    encoding2_path = input("Ruta del archivo .npy de encoding 2: ")
    
    if compare_face_encodings(encoding1_path, encoding2_path):
        print("Los rostros coinciden.")
    else:
        print("Los rostros no coinciden.")
