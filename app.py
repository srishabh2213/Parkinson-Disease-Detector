import base64
from flask import Flask, request, jsonify
import keras as ks
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import os
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = load_model('/Users/srivasr1/Desktop/Project/model.h5')

def preprocess_image(image, target_size):
    if image.mode != "L":  # Check if image is not grayscale
        image = image.convert("L")  # Convert to grayscale
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # Check if two files were sent
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    # Load the two images from the request
    file1 = request.files['file1']
    file2 = request.files['file2']

    # Preprocess the images
    image1 = Image.open(file1)
    image2 = Image.open(file2)
    processed_image1 = preprocess_image(image1, target_size=(128, 128))
    processed_image2 = preprocess_image(image2, target_size=(128, 128))

    # Predict
    prediction = model.predict([processed_image1, processed_image2]).tolist()

    # Create the response
    response = {
        'prediction': {
            'parkinson': prediction[0][0],
            'healthy': 1 - prediction[0][0]  # Assuming your model outputs the probability of parkinson
        }
    }
    return jsonify(response)
