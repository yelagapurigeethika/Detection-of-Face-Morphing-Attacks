import numpy as np
import cv2
from tensorflow.keras.models import load_model
from picamera2 import Picamera2


model = load_model("face_morph_detector.h5")

def preprocess_image(frame):
    """Preprocesses the Raspberry Pi camera frame for CNN prediction"""
    img = cv2.resize(frame, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)  
    return img

def predict_morph(frame):
    """Predicts if the captured frame is morphed or genuine"""
    img = preprocess_image(frame)
    prediction = model.predict(img)[0][0]
    return " Morphed Face" if prediction > 0.5 else " Genuine Face"



picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)  
picam2.preview_configuration.main.format = "RGB888"  
picam2.configure("preview")
picam2.start()

while True:
    frame = picam2.capture_array()  
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 

    
    result = predict_morph(frame)

    
    cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    cv2.imshow("Face Morph Detection - Raspberry Pi", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


picam2.stop()
cv2.destroyAllWindows()