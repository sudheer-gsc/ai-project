import cv2
import os

def capture_images(person_name):
    # Directory to save images
    data_dir = './data'
    person_path = os.path.join(data_dir, person_name)

    if not os.path.exists(person_path):
        os.makedirs(person_path)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("I will now take 20 pictures. Press ENTER when ready.")
    input()
    
    count = 0
    while count < 20:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{person_path}/{person_name}_{count}.jpg", face)
            print(f"Images Saved: {count}")
            if count >= 20:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"All images saved in the folder: {person_path}")

if __name__ == "__main__":
    person_name = input("Enter the name of the person: ")
    capture_images(person_name)
