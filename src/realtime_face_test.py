import cv2
import os

# Haar cascade dosya yolu
CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "haarcascade_frontalface_default.xml"
)

def main():
    if not os.path.exists(CASCADE_PATH):
        print(f"[ERROR] Haar cascade bulunamadı: {CASCADE_PATH}")
        return

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    cap = cv2.VideoCapture(0)  # laptop kamerası

    if not cap.isOpened():
        print("[ERROR] Kamera açılamadı.")
        return

    print("[INFO] Kamera başlatıldı. Çıkmak için 'q' tuşuna bas.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame okunamadı.")
            break

        # Griye çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Yüz tespiti
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50)
        )

        # Bulunan yüzleri çiz
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Yuz Tespiti Test", frame)

        # q'ya basınca çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
