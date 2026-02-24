import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# =========================================
# Flask Configuration
# =========================================
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# =========================================
# Load Trained Model
# =========================================
model = load_model("model/blood_cell_model.h5")

# ⚠️ MUST MATCH FOLDER NAMES EXACTLY
class_labels = [
    'Basophil',
    'eosinophil',
    'lymphocyte',
    'monocyte',
    'neutrophil',
    'others'   # <-- your Non Blood folder
]

# =========================================
# Routes
# =========================================
# ==============================
# Home Page
# ==============================
@app.route('/')
def home():
    return render_template("index.html")

# ==============================
# About Page
# ==============================
@app.route('/about')
def about():
    return render_template("about.html")

# ==============================
# Contact Page
# ==============================
@app.route('/contact')
def contact():
    return render_template("contact.html")

# ==============================
# Prediction Route
# ==============================


@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']

    if file.filename == '':
        return "No file selected"

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess Image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    predictions = model.predict(img_array)[0]

    max_index = np.argmax(predictions)
    max_cell = class_labels[max_index]
    max_value = float(predictions[max_index]) * 100

    results = {}

    # =========================================
    # IF OTHERS -> INVALID IMAGE
    # =========================================
    if max_cell == "others":

        for label in class_labels:
            results[label] = 0

        results["others"] = 100

        display_label = "Invalid Image (Not a Blood Cell)"
        max_value = 100

    else:
        for i in range(len(predictions)):
            results[class_labels[i]] = float(predictions[i]) * 100

        display_label = max_cell

    return render_template(
        "result.html",
        results=results,
        max_cell=display_label,
        max_value=max_value,
        image=filename
    )

# =========================================
# Run App
# =========================================
if __name__ == '__main__':
    app.run(debug=True)