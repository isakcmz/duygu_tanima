import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from config import (
    TRAIN_NPZ_PATH,
    VAL_NPZ_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_CHANNELS,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def load_data():
    print("[INFO] Veri yükleniyor...")

    train = np.load(TRAIN_NPZ_PATH)
    val = np.load(VAL_NPZ_PATH)

    X_train, y_train = train["X"], train["y"]
    X_val, y_val = val["X"], val["y"]

    print(f"[INFO] X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[INFO] X_val: {X_val.shape}, y_val: {y_val.shape}")

    # One-hot encode (7 sınıf)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=7)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=7)

    return X_train, y_train, X_val, y_val


def build_cnn_model():
    print("[INFO] CNN modeli oluşturuluyor...")

    model = models.Sequential()

    # 1. Conv bloğu
    model.add(layers.Conv2D(32, (3, 3), activation="relu",
                            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
    model.add(layers.MaxPooling2D((2, 2)))

    # 2. Conv bloğu
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # 3. Conv bloğu
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(7, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def train_model():
    X_train, y_train, X_val, y_val = load_data()

    model = build_cnn_model()

    # Modelleri buraya kaydedeceğiz
    os.makedirs("models", exist_ok=True)
    save_path = "models/cnn_baseline.h5"

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=save_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )
    ]

    print("[INFO] Eğitim başlıyor...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=64,
        callbacks=callbacks
    )

    print("[INFO] Eğitim tamamlandı!")
    print(f"[INFO] En iyi model kaydedildi: {save_path}")


if __name__ == "__main__":
    train_model()
