import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

# Model yolu
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "cnn_baseline.h5"
)

# Haar Cascade yolu
CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "haarcascade_frontalface_default.xml"
)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


class EmotionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Duygu Tanıma Uygulaması")
        self.window.geometry("900x700")

        # Model yükleme
        self.model = tf.keras.models.load_model(MODEL_PATH)

        # Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        # Kamera
        self.cap = cv2.VideoCapture(0)

        # UI bileşenleri
        self.label = Label(window)
        self.label.pack()

        self.emotion_label = Label(window, text="", font=("Arial", 24), fg="green")
        self.emotion_label.pack(pady=10)

        self.close_button = Button(window, text="Kapat", command=self.close_app,
                                   font=("Arial", 14), bg="red", fg="white")
        self.close_button.pack(pady=20)

        # Frame yenileme
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        emotion_text = ""

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_gray, (48, 48))
            face_norm = face_resized.reshape(1, 48, 48, 1) / 255.0

            preds = self.model.predict(face_norm, verbose=0)[0]
            emotion_idx = np.argmax(preds)
            emotion_text = EMOTIONS[emotion_idx]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Etiketi güncelle
        if emotion_text != "":
            self.emotion_label.config(text=emotion_text)

        # Tkinter için formatla
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        self.window.after(10, self.update_frame)

    def close_app(self):
        self.cap.release()
        self.window.destroy()


def main():
        window = tk.Tk()
        app = EmotionApp(window)
        window.mainloop()


if __name__ == "__main__":
    main()
