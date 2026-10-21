import tensorflow as tf
from src.config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, SEED

def make_datasets():
    # image_dataset_from_directory converts grayscale to RGB if color_mode='rgb'
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        color_mode="rgb",
        label_mode="categorical"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        color_mode="rgb",
        label_mode="categorical"
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=False,
        color_mode="rgb",
        label_mode="categorical"
    )

    # Basic augmentations
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    def aug_map(x, y):
        return aug(x), y

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(aug_map, num_parallel_calls=AUTOTUNE)\
                       .prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)
    test_ds  = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds
