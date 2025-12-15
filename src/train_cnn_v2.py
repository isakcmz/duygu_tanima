import os
import numpy as np
import tensorflow as tf

from src.config import (
    PROCESSED_DIR,
    MODEL_DIR,
    IMG_H,
    IMG_W,
    NUM_CLASSES
)

# =========================
# NPZ YÜKLEME
# =========================
def load_npz(path):
    data = np.load(path)
    return data["X"], data["y"]

# =========================
# CNN + DATA AUGMENTATION
# =========================
def build_cnn_v3(input_shape=(48, 48, 1), num_classes=7):
    inputs = tf.keras.Input(shape=input_shape)

    # 🔹 Data Augmentation (SADECE TRAIN SIRASINDA AKTİF)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
    ])

    x = data_augmentation(inputs)

    # =========================
    # Block 1
    # =========================
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # =========================
    # Block 2
    # =========================
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    # =========================
    # Block 3
    # =========================
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    # =========================
    # Head
    # =========================
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.50)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="cnn_v3_augmented")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# =========================
# TRAIN
# =========================
def main():
    train_path = os.path.join(PROCESSED_DIR, "fer2013_train.npz")
    val_path   = os.path.join(PROCESSED_DIR, "fer2013_val.npz")

    print("[INFO] NPZ yükleniyor...")
    X_train, y_train = load_npz(train_path)
    X_val, y_val = load_npz(val_path)

    print("[INFO] Model oluşturuluyor (CNN v3 + Data Augmentation)...")
    model = build_cnn_v3(input_shape=(IMG_H, IMG_W, 1), num_classes=NUM_CLASSES)

    os.makedirs(MODEL_DIR, exist_ok=True)

    ckpt_path = os.path.join(MODEL_DIR, "cnn_v3_best.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    print("[INFO] Eğitim başlıyor...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=64,
        callbacks=callbacks
    )

    final_path = os.path.join(MODEL_DIR, "cnn_v3_final.h5")
    model.save(final_path)

    print(f"[INFO] Kaydedildi: {final_path}")
    print(f"[INFO] En iyi model: {ckpt_path}")

if __name__ == "__main__":
    main()
