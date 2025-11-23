import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import customtkinter as ctk

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


class ModernEmotionApp:
    def __init__(self):
        # Tema ayarları
        ctk.set_appearance_mode("dark")          # "dark", "light", "system"
        ctk.set_default_color_theme("blue")      # "blue", "green", "dark-blue"

        # Ana pencere
        self.window = ctk.CTk()
        self.window.title("Duygu Tanıma Sistemi")
        self.window.geometry("1100x700")
        self.window.resizable(False, False)

        # Ana grid
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=3)
        self.window.grid_rowconfigure(0, weight=1)

        # Sol panel (sidebar)
        self.sidebar_frame = ctk.CTkFrame(self.window, corner_radius=15)
        self.sidebar_frame.grid(row=0, column=0, sticky="nswe", padx=15, pady=15)

        self.sidebar_frame.grid_rowconfigure(0, weight=0)
        self.sidebar_frame.grid_rowconfigure(1, weight=0)
        self.sidebar_frame.grid_rowconfigure(2, weight=0)
        self.sidebar_frame.grid_rowconfigure(3, weight=1)
        self.sidebar_frame.grid_rowconfigure(4, weight=0)

        self.title_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Duygu Tanıma",
            font=("Segoe UI", 26, "bold")
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        self.subtitle_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Webcam üzerinden\nanlık duygu analizi",
            font=("Segoe UI", 14)
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        # Duygu etiketi
        self.current_emotion_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Duygu: -",
            font=("Segoe UI", 22, "bold"),
            text_color="#00ff99"
        )
        self.current_emotion_label.grid(row=2, column=0, padx=20, pady=(10, 10), sticky="w")

        # Durum etiketi
        self.status_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Durum: Kamera başlatılıyor...",
            font=("Segoe UI", 13)
        )
        self.status_label.grid(row=3, column=0, padx=20, pady=(10, 10), sticky="nw")

        # Alt butonlar
        self.button_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.button_frame.grid(row=4, column=0, padx=20, pady=(10, 20), sticky="sew")
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)

        self.start_button = ctk.CTkButton(
            self.button_frame,
            text="Yeniden Başlat",
            command=self.restart_camera
        )
        self.start_button.grid(row=0, column=0, padx=(0, 10), pady=10, sticky="ew")

        self.close_button = ctk.CTkButton(
            self.button_frame,
            text="Kapat",
            fg_color="#ff4444",
            hover_color="#cc0000",
            command=self.close_app
        )
        self.close_button.grid(row=0, column=1, padx=(10, 0), pady=10, sticky="ew")

        # Sağ taraf: video alanı
        self.video_frame = ctk.CTkFrame(self.window, corner_radius=15)
        self.video_frame.grid(row=0, column=1, sticky="nswe", padx=(0, 15), pady=15)
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.video_frame, text="", corner_radius=10)
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Model ve kamera
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.status_label.configure(text="Durum: Kamera açılamadı!", text_color="red")
        else:
            self.status_label.configure(text="Durum: Çalışıyor", text_color="#00ff99")

        # Smoothing için basit liste
        self.pred_history = []
        self.smooth_window = 8  # son 8 frame'i ortala

        # Frame döngüsünü başlat
        self.update_frame()

        self.window.protocol("WM_DELETE_WINDOW", self.close_app)

    def restart_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.status_label.configure(text="Durum: Yeniden başlatıldı", text_color="#00ff99")
        else:
            self.status_label.configure(text="Durum: Kamera açılamadı", text_color="red")

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.window.after(50, self.update_frame)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.window.after(50, self.update_frame)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        emotion_text = "-"

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_gray, (48, 48))
            face_norm = face_resized.reshape(1, 48, 48, 1) / 255.0

            preds = self.model.predict(face_norm, verbose=0)[0]
            self.pred_history.append(preds)
            if len(self.pred_history) > self.smooth_window:
                self.pred_history.pop(0)

            avg_preds = np.mean(self.pred_history, axis=0)
            emotion_idx = int(np.argmax(avg_preds))
            emotion_text = EMOTIONS[emotion_idx]

            # Yüz kutusu
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Sol panelde duygu yazısı
        self.current_emotion_label.configure(text=f"Duygu: {emotion_text}")

        # BGR -> RGB -> PIL -> ImageTk
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        pil_image = pil_image.resize((800, 600))  # pencereye sığdır
        imgtk = ImageTk.PhotoImage(image=pil_image)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.window.after(30, self.update_frame)

    def close_app(self):
        if self.cap is not None:
            self.cap.release()
        self.window.destroy()


def main():
    app = ModernEmotionApp()
    app.window.mainloop()


if __name__ == "__main__":
    main()
