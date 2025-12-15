import os
import numpy as np
import tensorflow as tf

from src.config import (
    MODEL_DIR,
    TRAIN_NPZ_PATH,
    VAL_NPZ_PATH,
    IMG_H,
    IMG_W,
    NUM_CLASSES
)

# =========================
# Ayarlar
# =========================
SEQ_LEN = 10                  # 10 frame -> 1 sequence
SEQS_PER_CLASS_TRAIN = 2500   # her duygu için kaç sequence üretelim
SEQS_PER_CLASS_VAL   = 400

BASE_CNN_PATH = os.path.join(MODEL_DIR, "cnn_v3_best.h5")

# =========================
# Veri yükleme
# =========================
def load_npz(path):
    data = np.load(path)
    return data["X"], data["y"]

# =========================
# Sequence üretme (FER tek kare -> aynı sınıftan rastgele SEQ)
# =========================
def make_sequences(X, y, seq_len, seqs_per_class, num_classes, seed=42):
    rng = np.random.default_rng(seed)

    X_out = []
    y_out = []

    for cls in range(num_classes):
        idxs = np.where(y == cls)[0]
        if len(idxs) == 0:
            continue

        # her sequence için aynı sınıftan rastgele seq_len adet indeks seç
        for _ in range(seqs_per_class):
            chosen = rng.choice(idxs, size=seq_len, replace=True)
            X_out.append(X[chosen])   # (seq_len, 48, 48, 1)
            y_out.append(cls)

    X_out = np.array(X_out, dtype=np.float32)
    y_out = np.array(y_out, dtype=np.int64)

    # karıştır
    perm = rng.permutation(len(y_out))
    X_out = X_out[perm]
    y_out = y_out[perm]

    return X_out, y_out

# =========================
# CNN feature extractor (DONMUŞ)
# =========================
def build_feature_extractor(cnn_path):
    base = tf.keras.models.load_model(cnn_path)

    # Son katman softmax Dense, onun bir önceki katmanı (Dropout) 256-dim özellik verir
    feature_output = base.layers[-2].output
    extractor = tf.keras.Model(inputs=base.input, outputs=feature_output, name="cnn_feature_extractor")

    extractor.trainable = False
    for layer in extractor.layers:
        layer.trainable = False

    return extractor

# =========================
# CNN + LSTM model
# =========================
def build_cnn_lstm(feature_extractor, seq_len, num_classes):
    seq_input = tf.keras.Input(shape=(seq_len, IMG_H, IMG_W, 1), name="seq_input")

    # her frame -> CNN feature (256)
    x = tf.keras.layers.TimeDistributed(feature_extractor, name="td_cnn")(seq_input)

    # zaman bilgisi
    x = tf.keras.layers.LSTM(128, return_sequences=False, name="lstm")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(seq_input, out, name="cnn_lstm")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    # 1) NPZ yükle
    print("[INFO] NPZ yükleniyor...")
    X_train, y_train = load_npz(TRAIN_NPZ_PATH)
    X_val, y_val = load_npz(VAL_NPZ_PATH)

    print("[INFO] Train:", X_train.shape, y_train.shape)
    print("[INFO] Val  :", X_val.shape, y_val.shape)

    # 2) Sequence üret
    print("[INFO] Sequence üretiliyor...")
    Xs_train, ys_train = make_sequences(
        X_train, y_train, SEQ_LEN, SEQS_PER_CLASS_TRAIN, NUM_CLASSES, seed=42
    )
    Xs_val, ys_val = make_sequences(
        X_val, y_val, SEQ_LEN, SEQS_PER_CLASS_VAL, NUM_CLASSES, seed=123
    )

    print("[INFO] X_seq_train:", Xs_train.shape, "y:", ys_train.shape)
    print("[INFO] X_seq_val  :", Xs_val.shape, "y:", ys_val.shape)

    # 3) Feature extractor (CNN donmuş)
    if not os.path.exists(BASE_CNN_PATH):
        raise FileNotFoundError(f"CNN modeli bulunamadı: {BASE_CNN_PATH}")

    print("[INFO] Feature extractor yükleniyor:", BASE_CNN_PATH)
    extractor = build_feature_extractor(BASE_CNN_PATH)

    # 4) CNN+LSTM model
    print("[INFO] CNN+LSTM modeli oluşturuluyor...")
    model = build_cnn_lstm(extractor, SEQ_LEN, NUM_CLASSES)
    model.summary()

    # 5) Callbacks + Train
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_path = os.path.join(MODEL_DIR, "cnn_lstm_best.h5")
    final_path = os.path.join(MODEL_DIR, "cnn_lstm_final.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    print("[INFO] Eğitim başlıyor...")
    model.fit(
        Xs_train, ys_train,
        validation_data=(Xs_val, ys_val),
        epochs=25,
        batch_size=32,
        callbacks=callbacks
    )

    model.save(final_path)
    print("[INFO] Kaydedildi:", final_path)
    print("[INFO] En iyi model:", best_path)

if __name__ == "__main__":
    main()
