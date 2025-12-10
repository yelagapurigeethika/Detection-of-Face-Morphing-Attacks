import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained CNN model
model = load_model("face_morph_detector.h5")

# Load Haar cascade (use the correct path found from the steps above)
face_cascade = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

def preprocess_image(image):
    """Preprocesses an image for CNN prediction"""
    img = cv2.resize(image, (128, 128)) / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img

def predict_morph(image):
    """Predicts if the given image contains a morphed or genuine face"""
    img = preprocess_image(image)
    prediction = model.predict(img)[0][0]
    return "  Morphed Face" if prediction > 0.5 else " Genuine Face"

def detect_from_image(image_path):
    """Loads an image from a file path and predicts morph status"""
    img = cv2.imread(image_path)
    if img is None:
        print(f" Error: Cannot read image '{image_path}'")
        return
    result = predict_morph(img)
    print(f"Prediction: {result}")

image_path = input("Enter the image path: ").strip()
detect_from_image(image_path)
b 