import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque


# Model yolu
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "cnn_baseline.h5"
)

# Haar cascade yolu
CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "haarcascade_frontalface_default.xml"
)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def main():
    print("[INFO] Model yükleniyor...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("[INFO] Haar cascade yükleniyor...")
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # Son N tahmini saklamak için bir kuyruk
    SMOOTH_WINDOW = 10
    prob_history = deque(maxlen=SMOOTH_WINDOW)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Kamera açılamadı.")
        return

    print("[INFO] Başlatıldı. Çıkmak için Q.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_gray, (48, 48))
            face_norm = face_resized.reshape(1, 48, 48, 1) / 255.0

            preds = model.predict(face_norm, verbose=0)[0]  # shape: (7,)
            prob_history.append(preds)

            # Kuyruktaki tüm tahminlerin ortalamasını al
            avg_preds = np.mean(prob_history, axis=0)
            emotion_idx = np.argmax(avg_preds)
            emotion_text = EMOTIONS[emotion_idx]

            # Yüz kutusu
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Duygu yazısı
            cv2.putText(
                frame,
                emotion_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        cv2.imshow("Gerçek Zamanlı Duygu Tanıma", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
