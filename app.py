import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# -----------------------------
# Flask App Config
# -----------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "model", "blood_cell_model.h5")

# Create upload folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------------
# Load Trained Model
# -----------------------------
model = load_model(MODEL_PATH)

# ⚠️ ORDER MUST MATCH TRAINING
class_names = [
    "Eosinophil",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Others"
]

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]

    if file.filename == "":
        return redirect(url_for("index"))

    # Safe filename (FIXED LINE ✅)
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    # Save image
    file.save(filepath)

    # -----------------------------
    # Image Preprocessing
    # -----------------------------
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # Prediction
    # -----------------------------
    predictions = model.predict(img_array)[0]

    results = {
    "Neutrophil": float(round(predictions[3] * 100, 2)),
    "Lymphocyte": float(round(predictions[1] * 100, 2)),
    "Monocyte": float(round(predictions[2] * 100, 2)),
    "Eosinophil": float(round(predictions[0] * 100, 2))
}


    # Dominant cell
    max_cell = max(results, key=results.get)
    max_value = results[max_cell]

    # If model predicts "Others"
    predicted_index = np.argmax(predictions)
    if class_names[predicted_index] == "Others":
        max_cell = "Invalid Image (Not a Blood Cell)"
        max_value = round(predictions[predicted_index] * 100, 2)

    # -----------------------------
    # Send to Result Page
    # -----------------------------
    return render_template(
        "result.html",
        image=filename,
        results=results,
        max_cell=max_cell,
        max_value=max_value
    )


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
