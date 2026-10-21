import os
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
from src.config import (IMG_SIZE, DROPOUT, BASE_LR, EPOCHS, FINE_TUNE_EPOCHS,
                        MODEL_DIR, CKPT_DIR, LOG_DIR, CLASSES, LABEL_SMOOTHING)
from src.data_prep import make_datasets

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def build_model(num_classes=4):
    base = EfficientNetV2L(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,),
        pooling="avg",                # global average pooling
        include_preprocessing=True    # feed [0,255] floats
    )
    base.trainable = False  # stage 1: freeze

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    opt = tf.keras.optimizers.Adam(learning_rate=BASE_LR)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy",
                 tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2")]
    )
    return model, base

def main():
    train_ds, val_ds, test_ds = make_datasets()
    model, base = build_model(num_classes=len(CLASSES))

    cbs = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CKPT_DIR, "stage1_best.keras"),
            monitor="val_accuracy", save_best_only=True
        ),
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.2, patience=2),
        callbacks.TensorBoard(LOG_DIR)
    ]

    # Stage 1: train head
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs)

    # Stage 2: fine-tune top layers of base
    base.trainable = True
    # unfreeze top N blocks (lighter fine-tuning)
    for layer in base.layers[:-60]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LR * 0.2),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2")]
    )

    cbs_ft = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(CKPT_DIR, "stage2_best.keras"),
            monitor="val_accuracy", save_best_only=True
        ),
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.2, patience=2),
        callbacks.TensorBoard(LOG_DIR)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, callbacks=cbs_ft)

    # Save final
    model.save(os.path.join(MODEL_DIR, "effv2l_alzheimers.h5"))
    # quick test accuracy
    print("Test:", model.evaluate(test_ds, verbose=0))

if __name__ == "__main__":
    main()
