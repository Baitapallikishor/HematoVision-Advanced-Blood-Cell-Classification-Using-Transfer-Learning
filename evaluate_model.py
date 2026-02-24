import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# Configuration
# ===============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

val_dir = "dataset/validation"
MODEL_PATH = "model/blood_cell_model.h5"

# ===============================
# Load Model
# ===============================
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

# ===============================
# Validation Data
# ===============================
val_gen = ImageDataGenerator(rescale=1./255)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ===============================
# Predictions
# ===============================
val_data.reset()
predictions = model.predict(val_data, verbose=1)

y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

# ===============================
# Confusion Matrix
# ===============================
cm = confusion_matrix(y_true, y_pred)
class_names = list(val_data.class_indices.keys())

plt.figure(figsize=(6,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix")
plt.show()

# ===============================
# Classification Report
# ===============================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))