import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import *

LABELS = ['with_mask', 'without_mask']
COLORS = [GREEN, RED]

def detect_live():
    if not __import__('os').path.exists(MODEL_PATH):
        print("Model nahi mila — pehle train_model.py chalao")
        return

    model = load_model(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Camera nahi khula")
        return

    print("Live detection shuru — 'q' dabao band karne ke liye")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(50,50))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face_norm = face_resized / 255.0
            face_input = np.expand_dims(face_norm, axis=0)

            prediction = model.predict(face_input, verbose=0)
            with_mask_conf = prediction[0][0]

            if with_mask_conf > 0.35:
                idx = 0
            else:
                idx = np.argmax(prediction)

            label = LABELS[idx]
            confidence = prediction[0][idx]
            color = COLORS[idx]

            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            text = f"{label}: {confidence*100:.1f}%"
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Face Mask Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_live()