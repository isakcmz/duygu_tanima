import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from PIL import Image, ImageTk
import customtkinter as ctk

# =========================
# PATHLER
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_lstm_best.h5")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

# =========================
# SABİTLER
# =========================
IMG_H, IMG_W = 48, 48
SEQ_LEN = 10

EMOTIONS = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

# =========================
# APP
# =========================
class EmotionAppLSTM:
    def __init__(self):
        # UI theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.window = ctk.CTk()
        self.window.title("Duygu Tanıma Sistemi (CNN + LSTM)")
        self.window.geometry("1100x700")
        self.window.resizable(True, True)

        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=3)
        self.window.grid_rowconfigure(0, weight=1)

        # =========================
        # SOL PANEL
        # =========================
        self.sidebar = ctk.CTkFrame(self.window, corner_radius=15)
        self.sidebar.grid(row=0, column=0, sticky="nswe", padx=15, pady=15)

        self.title_label = ctk.CTkLabel(
            self.sidebar,
            text="Duygu Tanıma",
            font=("Segoe UI", 26, "bold")
        )
        self.title_label.pack(padx=20, pady=(20, 10), anchor="w")

        self.subtitle = ctk.CTkLabel(
            self.sidebar,
            text="CNN + LSTM\nRealtime analiz",
            font=("Segoe UI", 14)
        )
        self.subtitle.pack(padx=20, pady=(0, 20), anchor="w")

        self.emotion_box = ctk.CTkFrame(self.sidebar, corner_radius=12)
        self.emotion_box.pack(padx=20, pady=20, fill="x")

        self.emotion_label = ctk.CTkLabel(
            self.emotion_box,
            text="—",
            font=("Segoe UI", 28, "bold")
        )
        self.emotion_label.pack(pady=20)

        self.status_label = ctk.CTkLabel(
            self.sidebar,
            text="Durum: Başlatılıyor...",
            font=("Segoe UI", 13)
        )
        self.status_label.pack(padx=20, pady=(10, 0), anchor="w")

        self.close_btn = ctk.CTkButton(
            self.sidebar,
            text="Kapat",
            fg_color="#aa3333",
            hover_color="#882222",
            command=self.close_app
        )
        self.close_btn.pack(padx=20, pady=20, fill="x")

        # =========================
        # SAĞ PANEL (VIDEO)
        # =========================
        self.video_frame = ctk.CTkFrame(self.window, corner_radius=15)
        self.video_frame.grid(row=0, column=1, sticky="nswe", padx=(0, 15), pady=15)
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # =========================
        # MODEL + CAMERA
        # =========================
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.configure(text="Durum: Kamera açılamadı!")
        else:
            self.status_label.configure(text="Durum: Çalışıyor")

        # =========================
        # LSTM BUFFER
        # =========================
        self.frame_buffer = deque(maxlen=SEQ_LEN)
        self.last_emotion = "—"

        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.close_app)

    # =========================
    # FRAME LOOP
    # =========================
    def update_frame(self):
        if not self.cap.isOpened():
            self.window.after(50, self.update_frame)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.window.after(50, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        emotion_text = self.last_emotion

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_W, IMG_H))
            face = face / 255.0
            face = face.reshape(IMG_H, IMG_W, 1)

            self.frame_buffer.append(face)

            if len(self.frame_buffer) == SEQ_LEN:
                seq = np.array(self.frame_buffer)[np.newaxis, ...]
                preds = self.model.predict(seq, verbose=0)[0]
                idx = int(np.argmax(preds))
                emotion_text = EMOTIONS[idx]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if emotion_text != self.last_emotion:
            self.last_emotion = emotion_text
            self.emotion_label.configure(text=emotion_text)

        # Video göster
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((800, 600))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.window.after(30, self.update_frame)

    def close_app(self):
        if self.cap:
            self.cap.release()
        self.window.destroy()

# =========================
# MAIN
# =========================
def main():
    app = EmotionAppLSTM()
    app.window.mainloop()

if __name__ == "__main__":
    main()
