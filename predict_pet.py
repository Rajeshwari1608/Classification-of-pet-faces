import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model and class names
model = tf.keras.models.load_model('oxford_pet_model.h5')

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Predict an image
img_path = 'test.jpg'  # ← Replace with your test image
img = image.load_img(img_path, target_size=(160, 160))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print("Predicted pet breed:", predicted_class)
 