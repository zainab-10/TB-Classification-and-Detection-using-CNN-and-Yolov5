# app.py
from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from tensorflow import keras
import torch
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='.', static_url_path='/static')
port = 5000

# Load the classification model
model_path = 'resnet_1.h5'
classification_model = keras.models.load_model(model_path)
class_labels = ["Healthy", "Sick", "TB"]

# Load the YOLOv5 object detection model
yolo_model_path = 'last.pt'
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/form', methods=['GET'])
def render_form():
    return render_template('form.html')

@app.route('/process_image', methods=['GET','POST'])
def process_image():
    # Get the uploaded image
    image_path = request.files['image'].read()
    image_path = Image.open(BytesIO(image_path))
    image_path.save('uploaded_image.jpg')  # Save the image if needed

    # Preprocess the image for classification
    image = image_path.resize((224, 224))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Predict the class
    preds = classification_model.predict(image)
    predicted_class_index = np.argmax(preds)
    predicted_class = class_labels[predicted_class_index]

    if predicted_class_index == 2:
        # Preprocess the image for object detection
        detection_results = yolo_model(image_path)
        # Get the image with detected objects
        detection_image = detection_results.render()[0]

        # Save the detected image using matplotlib
        plt.figure()
        plt.imshow(detection_image)
        plt.axis('off')
        plt.savefig('detection_image.png')
        plt.close()
        return render_template('detection_result.html', predicted_class=predicted_class)
    else:
        return render_template('classification_result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
