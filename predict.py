# import tensorflow as tf
# import cv2
# import numpy as np

# # Load trained model
# model = tf.keras.models.load_model('model/footwear_safety_model.h5')

# def predict_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (150, 150))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)
#     pred = model.predict(img)[0][0]
#     return "Safe" if pred < 0.5 else "Unsafe"

# # Test example
# img_path = 'dataset/safe/safe1.jpg'  # Change this as needed
# print("Prediction:", predict_image(img_path))

import tensorflow as tf
import cv2
import numpy as np
import os

# Class labels (must match training folder names)
CLASS_NAMES = ['replace_soon', 'safe', 'unsafe']  # alphabetical order by default

# Load trained model
# model_path = 'model/footwear_safety_model.h5'
# assert os.path.exists(model_path), f"Model not found: {model_path}"
# model = tf.keras.models.load_model(model_path)

# def predict_image(image_path):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image not found: {image_path}")
    
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (150, 150))
#     img = img.astype('float32') / 255.0
#     img = np.expand_dims(img, axis=0)

#     predictions = model.predict(img, verbose=0)[0]
#     class_index = np.argmax(predictions)
#     confidence = predictions[class_index] * 100
#     label = CLASS_NAMES[class_index]
    
#     return label, confidence

# Load the model once (for efficiency)


# Test example
# if __name__ == "__main__":
#     test_image_path = 'footwear_dataset/safe/safe_1.jpg'  
#     result, confidence = predict_image(test_image_path)
#     print(f"Prediction: {result} ({confidence:.2f}% confidence)")


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model once
model = load_model("model/footwear_safety_model.h5")

# Match the class order from training
class_labels = ['replace_soon', 'safe', 'unsafe']

# def predict_image(image_path):
#     # Load and preprocess image
#     img = image.load_img(image_path, target_size=(150, 150))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     # Predict
#     predictions = model.predict(img_array, verbose=0)[0]  # shape: (3,)
#     class_probs = dict(zip(class_labels, predictions))
#     predicted_label = class_labels[np.argmax(predictions)]

#     return predicted_label, class_probs

def predict_image(image_path):
    # Open image with Pillow
    img = Image.open(image_path)

    # Convert transparent or palette images to RGB
    if img.mode in ('P', 'RGBA', 'LA'):
        img = img.convert('RGB')

    # Resize and convert to array
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]  # shape: (3,)
    class_probs = dict(zip(class_labels, predictions))
    predicted_label = class_labels[np.argmax(predictions)]

    return predicted_label, class_probs