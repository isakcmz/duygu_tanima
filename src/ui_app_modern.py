import os
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from PIL import Image, ImageTk
import customtkinter as ctk
import csv
from datetime import datetime


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "cnn_lstm_best.h5"
)

CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "haarcascade_frontalface_default.xml"
)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


class ModernEmotionApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ANA PENCERE
        self.window = ctk.CTk()
        self.window.title("Duygu Tanıma Sistemi")
        self.window.state("zoomed")
        self.window.resizable(True, True)

        # GRID
        self.window.grid_columnconfigure(0, weight=1)
        self.window.grid_columnconfigure(1, weight=4)
        self.window.grid_rowconfigure(0, weight=1)


        # ================== LOG DOSYASI AYARI ==================
        # logs/emotion_log.csv dosyasını hazırlıyoruz
        self.log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs",
            "emotion_log.csv"
        )

        # logs klasörü yoksa oluştur
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Uygulama her açıldığında log dosyasını sıfırdan başlat
        # Log dosyasının yolu
        self.log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs",
            "emotion_log.csv"
        )
        # Zaman takip değişkeni (her saniyeyi kontrol etmek için)
        self.last_logged_second = None

        # klasör yoksa oluştur
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Seans başlangıcını dosyaya yaz
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== SESSION START {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

        # =======================================================



        # =====================================================================
        # SOL PANEL (SABİT, KAYMAYAN, MODERN)
        # =====================================================================
        self.sidebar_frame = ctk.CTkFrame(self.window, corner_radius=15, fg_color="#111111")
        self.sidebar_frame.grid(row=0, column=0, sticky="nswe", padx=15, pady=15)
        self.sidebar_frame.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Duygu Tanıma",
            font=("Segoe UI", 26, "bold"),
            text_color="white"
        )
        self.title_label.grid(row=0, column=0, pady=(20, 10))

        self.subtitle_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Webcam üzerinden\nanlık duygu analizi",
            font=("Segoe UI", 14),
            text_color="#bbbbbb"
        )
        self.subtitle_label.grid(row=1, column=0, pady=(0, 20))

        # Daha modern butonlar:
        self.start_button = ctk.CTkButton(
            self.sidebar_frame,
            text="📷 Yeniden Başlat",
            font=("Segoe UI", 15),
            fg_color="#0A84FF",
            hover_color="#0061d5",
            corner_radius=10,
            command=self.restart_camera
        )
        self.start_button.grid(row=2, column=0, pady=10, padx=30, sticky="ew")

        self.close_button = ctk.CTkButton(
            self.sidebar_frame,
            text="❌ Kapat",
            font=("Segoe UI", 15),
            fg_color="#ff3b31",
            hover_color="#cc2b25",
            corner_radius=10,
            command=self.close_app
        )
        self.close_button.grid(row=3, column=0, pady=10, padx=30, sticky="ew")

        # =====================================================================
        # SAĞ PANEL (KAMERA + ALTINDA DUYGU)
        # =====================================================================
        self.video_frame = ctk.CTkFrame(self.window, corner_radius=15, fg_color="#111111")
        self.video_frame.grid(row=0, column=1, sticky="nswe", padx=(0, 15), pady=15)

        self.video_frame.grid_rowconfigure(0, weight=8)
        self.video_frame.grid_rowconfigure(1, weight=2)
        self.video_frame.grid_columnconfigure(0, weight=1)

        # Kamera görüntüsü alanı
        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="",
            fg_color="#1a1a1a"
        )
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))

        # Kameranın ALTINDA büyük duygu yazısı
        self.big_emotion_label = ctk.CTkLabel(
            self.video_frame,
            text="DUYGU: -",
            font=("Segoe UI", 32, "bold"),
            fg_color="#1c1c1e",       # Apple tarzı koyu gri chip
            text_color="#ffffff",
            corner_radius=12,
            padx=25,
            pady=15
        )
        self.big_emotion_label.grid(row=1, column=0, pady=(20, 30))

        # MODEL + CAMERA
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        self.cap = cv2.VideoCapture(0)

        # LSTM için 10 frame buffer
        self.frame_buffer = deque(maxlen=10)


        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.close_app)

    # =====================================================================
    # CAMERA RESTART
    # =====================================================================
    def restart_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.frame_buffer.clear()

    # =====================================================================
    # FRAME UPDATE LOOP
    # =====================================================================
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
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face_norm = face.reshape(48, 48, 1) / 255.0

            # buffer'a ekle
            self.frame_buffer.append(face_norm)

            # 10 frame dolunca LSTM ile tahmin yap
            if len(self.frame_buffer) == 10:
                seq = np.array(self.frame_buffer).reshape(1, 10, 48, 48, 1)
                preds = self.model.predict(seq, verbose=0)[0]
                emotion_idx = int(np.argmax(preds))
                emotion_text = EMOTIONS[emotion_idx]
            else:
                emotion_text = "-"


            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 150), 2)


                # === LOG'A YAZ (yüz bulunduysa) ===
        # === HER SANİYEDE TEK SATIR LOG YAZ ===
        current_second = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.last_logged_second != current_second:
            self.last_logged_second = current_second

            if emotion_text != "-":
                with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([current_second, emotion_text])




        # DUYGUYU kameranın altına yaz
        self.big_emotion_label.configure(text=f"DUYGU: {emotion_text.upper()}")

        # KAMERA GÖRÜNTÜSÜ
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = pil.resize((900, 600))
        imgtk = ImageTk.PhotoImage(pil)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.window.after(15, self.update_frame)

    def close_app(self):
        if self.cap:
            self.cap.release()

            # =============== OTURUM ÖZETİ ===============
        stats = {}
        total = 0

        # CSV'yi oku
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if "," in line and "SESSION" not in line:
                    _, emotion = line.strip().split(",")
                    stats[emotion] = stats.get(emotion, 0) + 1
                    total += 1

        # Seansı bitiş tarihi ile yaz
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"=== SESSION END {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

            if total == 0:
                f.write("No emotion data recorded.\n")
            else:
                for emo, count in stats.items():
                    percent = (count / total) * 100
                    f.write(f"{emo}: %{percent:.1f}\n")

            f.write("-----------------------------------------\n\n")

        # Uygulamayı kapat
        self.window.destroy()


def main():
    app = ModernEmotionApp()
    app.window.mainloop()


if __name__ == "__main__":
    main()
