import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("outputs/models/lung_model.keras")

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/lung_dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224,224),
    batch_size=16
)

y_true = []
y_pred = []

for images, labels in val_ds:

    preds = model.predict(images)

    y_true.extend(labels.numpy())
    y_pred.extend(preds.argmax(axis=1))

print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True)

plt.savefig("outputs/plots/confusion_matrix.png")