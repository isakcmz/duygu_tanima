import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

FER_CSV_PATH = os.path.join(RAW_DIR, "fer2013.csv")

TRAIN_NPZ_PATH = os.path.join(PROCESSED_DIR, "fer2013_train.npz")
VAL_NPZ_PATH = os.path.join(PROCESSED_DIR, "fer2013_val.npz")
TEST_NPZ_PATH = os.path.join(PROCESSED_DIR, "fer2013_test.npz")

IMG_HEIGHT = 48
IMG_WIDTH = 48
IMG_CHANNELS = 1

EMOTION_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}
