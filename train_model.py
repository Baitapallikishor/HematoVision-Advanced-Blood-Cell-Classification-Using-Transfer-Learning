import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# ===============================
# Configuration
# ===============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

train_dir = "dataset/train"
val_dir = "dataset/validation"

# ===============================
# Data Generators
# ===============================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("Class Indices:", train_data.class_indices)

# ===============================
# Load Base Model
# ===============================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = True

for layer in base_model.layers[:100]:
    layer.trainable = False

# ===============================
# Custom Layers
# ===============================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)

# 🔥 6 CLASSES
output = Dense(6, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# Train
# ===============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ===============================
# Save Model
# ===============================
model.save("model/blood_cell_model.h5")
print("Model Saved Successfully")
print("Output Shape:", model.output_shape)