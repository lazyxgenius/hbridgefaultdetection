import setuptools
import tensorflow as tf
import keras

import joblib
import numpy as np
from tensorflow.keras.preprocessing import image

# Load and preprocess the image you want to predict
image_path = r"C:\Users\Aditya PC\PycharmProjects\hbridgefaultdetection\New dataset\Validation\fault_3_a\Fault_3_A_84.jpg"
img = image.load_img(image_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 225.0  # Normalize the image data

# Make a prediction
model = joblib.load('saved_hbridge.joblib')
prediction = model.predict(img_array)

# Get the class index with the highest probability
predicted_class_index = np.argmax(prediction)

# Define a dictionary that maps class indices to class labels
class_labels = {
    0: 'no fault',
    1: 'fault1a',
    2:'fault1b',
    3:'fault2a',
    4:'fault2b',
    5:'fault3a',
    6:'fault3b',
    7:'fault4a',
    8:'fault4b'
}

# Get the predicted class label
predicted_class_label = class_labels[predicted_class_index]

print("Predicted class:", predicted_class_label)
print("Class probabilities:", prediction)
