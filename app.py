import os
import numpy as np
import cv2
from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

# Flask app setup
app = Flask(__name__)

# Constants
MODEL_PATH = 'mobilenetv2_footprint.h5'
IMG_SIZE = 128
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = load_model(MODEL_PATH)

# Dynamically load class labels
train_dir = 'dataset/train'
CLASS_LABELS = os.listdir(train_dir)

def preprocess_image(image_path):
    """Preprocess the image for prediction."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction from uploaded image."""
    if 'file' not in request.files:
        return 'No file uploaded.', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected.', 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Preprocess and predict
    img = preprocess_image(file_path)
    predictions = model.predict(img)
    confidence_scores = predictions[0]
    predicted_index = np.argmax(confidence_scores)
    confidence = confidence_scores[predicted_index] * 100
    predicted_label = CLASS_LABELS[predicted_index]

    # Render result page
    return render_template('result.html',
                           prediction=predicted_label,
                           confidence=f"{confidence:.2f}%",
                           uploaded_image=file_path)

@app.route('/uploads/<filename>')
def serve_file(filename):
    """Serve uploaded files like images."""
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True)
