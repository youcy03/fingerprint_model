from flask import Flask, render_template, request
from utils import predict_fingerprint
import os
from werkzeug.utils import secure_filename
import time
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model and encoder once
model = load_model("model/final_good_model.keras")
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Preprocessing function (mimic utils if you want)
def preprocess_image(image_path):
    img = cv2.imread(image_path)  
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)  # shape: (1, 64, 64, 3)
    return img




@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.abspath(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            # Ensure upload directory exists
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            file.save(filepath)

            # Prediction
            img = preprocess_image(filepath)
            prediction_array = model.predict(img)
            confidence = float(np.max(prediction_array))
            predicted_index = np.argmax(prediction_array)

            if confidence < 0.9:
                predicted_label = "Unknown"
            else:
                predicted_label = label_encoder.inverse_transform([predicted_index])[0]

            confidence_percent = round(confidence * 100, 2)
            prediction = f"Predicted Label: {predicted_label} (Confidence: {confidence_percent}%)"
            image_path = "static/uploads/" + filename

            time.sleep(2.5)  # Small delay for visual feedback

    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
