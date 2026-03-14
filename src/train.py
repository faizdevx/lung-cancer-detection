from dataset import load_datasets
from model import build_model
import tensorflow as tf
import os

DATA_PATH = "data/lung_dataset"

train_ds, val_ds = load_datasets(DATA_PATH)

model = build_model()

os.makedirs("outputs/models", exist_ok=True)

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "outputs/models/lung_model.keras",
        save_best_only=True
    )
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)