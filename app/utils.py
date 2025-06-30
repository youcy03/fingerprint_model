import tensorflow as tf
import numpy as np
import cv2
import pickle
from sklearn.preprocessing import LabelEncoder


# Load your trained model
model = tf.keras.models.load_model("model/final_good_model.keras")

# Load the label encoder
with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Function to predict fingerprint
def predict_fingerprint(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    result_index = np.argmax(prediction)
    label = label_encoder.inverse_transform([result_index])[0]
    return f"Predicted Label: {label}"
