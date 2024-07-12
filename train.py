import os
import cv2
import numpy as np
from PIL import Image

def get_images_and_labels(data_dir):
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
    face_samples = []
    ids = []

    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        img_np = np.array(img, 'uint8')
        # Assume all images in the directory belong to the same person
        person_id = 1
        face_samples.append(img_np)
        ids.append(person_id)

    return face_samples, ids

def train_model():
    data_dir = './data'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)
        if os.path.isdir(person_path):
            print(f"Training on images of {person_name}...")
            faces, ids = get_images_and_labels(person_path)
            recognizer.train(faces, np.array(ids))
    
    recognizer.save('trainer.yml')
    print("Model trained and saved as 'trainer.yml'.")

if __name__ == "__main__":
    train_model()
