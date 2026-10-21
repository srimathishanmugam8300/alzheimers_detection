import tensorflow as tf
import numpy as np
import sys
from src.config import IMG_SIZE, CLASSES, MODEL_DIR

def predict(img_path):
    model = tf.keras.models.load_model(f"{MODEL_DIR}/effv2l_alzheimers.h5")
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE, color_mode="rgb")
    x = tf.keras.utils.img_to_array(img)  # float32 [0..255]
    x = np.expand_dims(x, 0)              # (1,H,W,3) ; preprocessing is inside model
    prob = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(prob))
    return CLASSES[idx], float(prob[idx]), prob

if __name__ == "__main__":
    pth = sys.argv[1]
    label, conf, prob = predict(pth)
    print(f"Pred: {label} ({conf:.3f})")
