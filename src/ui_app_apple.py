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

# Haar Cascade
CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "haarcascade_frontalface_default.xml"
)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


class AppleEmotionApp:
    def __init__(self):
        # Apple tarzı açık mod
        ctk.set_appearance_mode("light")

        # Pencere
        self.window = ctk.CTk()
        self.window.title("Emotion Recognition – Apple Style")
        self.window.state("zoomed")
        self.window.resizable(True, True)

        # Klasik Apple açık gri arka plan
        self.window.configure(bg="#f2f2f7")

        # Grid
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=4)
        self.window.grid_rowconfigure(0, weight=1)

        # ==== SOL PANEL (Apple Sidebar) ====
        self.sidebar = ctk.CTkFrame(
            self.window,
            corner_radius=20,
            fg_color="#ffffff"
        )
        self.sidebar.grid(row=0, column=0, sticky="nswe", padx=15, pady=15)

        self.sidebar.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.sidebar,
            text="Emotion AI",
            font=("Segoe UI", 28, "bold"),
            text_color="#111111"
        )
        self.title_label.grid(row=0, column=0, pady=(30, 10))

        self.info_label = ctk.CTkLabel(
            self.sidebar,
            text="Real-time emotion detection\npowered by AI",
            font=("Segoe UI", 15),
            text_color="#6e6e73"
        )
        self.info_label.grid(row=1, column=0, pady=(0, 30))

        # Apple-style chip
        self.emotion_chip = ctk.CTkLabel(
            self.sidebar,
            text="Emotion: -",
            font=("Segoe UI", 20, "bold"),
            text_color="#111111",
            fg_color="#e5e5ea",
            corner_radius=12,
            padx=20,
            pady=12
        )
        self.emotion_chip.configure(width=250)
        self.emotion_chip.configure(anchor="center")
        self.emotion_chip.configure(height=50)
        self.emotion_chip.grid(row=2, column=0, pady=20, padx=20, sticky="ew")

        # Butonlar (Apple pastel)
        self.restart_button = ctk.CTkButton(
            self.sidebar,
            text="Restart Camera",
            fg_color="#0a6b20",
            hover_color="#078018"
        )
        self.restart_button.grid(row=3, column=0, pady=(30, 10), padx=40, sticky="ew")
        self.restart_button.configure(command=self.restart_camera)

        self.close_button = ctk.CTkButton(
            self.sidebar,
            text="Close",
            fg_color="#fb3329",
            hover_color="#d3352a"
        )
        self.close_button.grid(row=4, column=0, pady=(5, 20), padx=40, sticky="ew")
        self.close_button.configure(command=self.close_app)

        # ==== SAĞ PANEL (Video Container) ====
        self.video_frame = ctk.CTkFrame(
            self.window,
            corner_radius=20,
            fg_color="#ffffff"
        )
        self.video_frame.grid(row=0, column=1, sticky="nswe", padx=(0, 15), pady=15)
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.video_frame, text="", fg_color="#ffffff")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Model, yüz algılayıcı ve kamera
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        self.cap = cv2.VideoCapture(0)

        # Smoothing
        self.pred_history = []
        self.smooth_window = 8

        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.close_app)

    def restart_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.window.after(20, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        emotion_text = "-"

        for (x, y, w, h) in faces:
            face_gray = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_gray, (48, 48))
            face_norm = face_resized.reshape(1, 48, 48, 1) / 255.0

            preds = self.model.predict(face_norm, verbose=0)[0]

            self.pred_history.append(preds)
            if len(self.pred_history) > self.smooth_window:
                self.pred_history.pop(0)

            avg_preds = np.mean(self.pred_history, axis=0)
            emotion_idx = int(np.argmax(avg_preds))
            emotion_text = EMOTIONS[emotion_idx]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (22, 160, 133), 2)

        self.emotion_chip.configure(text=f"Emotion: {emotion_text}")

        # Görüntüyü ekrana aktar
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((1200, 750))
        imgtk = ImageTk.PhotoImage(pil_img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.window.after(10, self.update_frame)

    def close_app(self):
        if self.cap:
            self.cap.release()
        self.window.destroy()


def main():
    app = AppleEmotionApp()
    app.window.mainloop()


if __name__ == "__main__":
    main()
