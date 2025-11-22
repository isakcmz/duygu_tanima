import os
import numpy as np
import pandas as pd

from src.config import (
    FER_CSV_PATH,
    TRAIN_NPZ_PATH,
    VAL_NPZ_PATH,
    TEST_NPZ_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS,
)


def load_fer2013_csv():
    """FER-2013 CSV dosyasını pandas ile okur."""
    if not os.path.exists(FER_CSV_PATH):
        raise FileNotFoundError(f"FER-2013 CSV bulunamadı: {FER_CSV_PATH}")

    print(f"[INFO] FER-2013 CSV okunuyor: {FER_CSV_PATH}")
    df = pd.read_csv(FER_CSV_PATH)
    print(f"[INFO] Toplam satır sayısı: {len(df)}")
    return df


def pixels_to_image_array(pixels_str):
    """
    '70 80 90 ...' gibi string'i alır,
    48x48x1 numpy array'e çevirir.
    """
    pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=' ')
    img = pixels.reshape(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    return img


def preprocess_fer2013(df):
    """
    Train / Val / Test setlerini oluşturur.
    """

    train_df = df[df["Usage"] == "Training"]
    val_df = df[df["Usage"] == "PublicTest"]
    test_df = df[df["Usage"] == "PrivateTest"]

    def df_to_xy(sub_df, name=""):
        print(f"[INFO] {name} seti işleniyor. Satır sayısı: {len(sub_df)}")
        X_list = []
        y_list = []

        for i, row in sub_df.iterrows():
            img = pixels_to_image_array(row["pixels"])
            label = int(row["emotion"])

            X_list.append(img)
            y_list.append(label)

        X = np.stack(X_list, axis=0).astype("float32")
        y = np.array(y_list, dtype=np.int64)

        # Normalize
        X /= 255.0

        print(f"[INFO] {name} X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    X_train, y_train = df_to_xy(train_df, "TRAIN")
    X_val, y_val = df_to_xy(val_df, "VAL")
    X_test, y_test = df_to_xy(test_df, "TEST")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def save_npz(X, Y, path):
    """X, Y arraylerini .npz formatında kaydeder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X, y=Y)
    print(f"[INFO] Kaydedildi: {path}")


def main():
    df = load_fer2013_csv()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocess_fer2013(df)

    save_npz(X_train, y_train, TRAIN_NPZ_PATH)
    save_npz(X_val, y_val, VAL_NPZ_PATH)
    save_npz(X_test, y_test, TEST_NPZ_PATH)

    print("[INFO] Tüm veri setleri başarıyla işlendi!")


if __name__ == "__main__":
    main()
