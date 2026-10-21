import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.config import TEST_DIR, IMG_SIZE, CLASSES, MODEL_DIR, SEED
import itertools

def load_test_ds():
    ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=32,
        seed=SEED,
        shuffle=False,
        color_mode="rgb",
        label_mode="categorical"
    )
    return ds

def main():
    ds = load_test_ds()
    model = tf.keras.models.load_model(f"{MODEL_DIR}/effv2l_alzheimers.h5")
    y_true = np.concatenate([y.numpy() for _, y in ds], axis=0)
    y_true = y_true.argmax(axis=1)

    y_pred_prob = model.predict(ds, verbose=0)
    y_pred = y_pred_prob.argmax(axis=1)

    print(classification_report(y_true, y_pred, target_names=CLASSES))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()
