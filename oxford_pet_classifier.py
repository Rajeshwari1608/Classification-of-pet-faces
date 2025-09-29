import tensorflow as tf
import matplotlib.pyplot as plt
import os
import re
import shutil
import pathlib
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Step 1: Reorganize Images into Class Folders (do this once)
base_dir = 'dataset/images'
organized_dir = 'dataset/organized'

if not os.path.exists(organized_dir):
    os.makedirs(organized_dir)

    # Group images by class using filename
    for img_name in os.listdir(base_dir):
        if img_name.endswith(".jpg"):
            breed = "_".join(img_name.lower().split("_")[:-1])
            class_dir = os.path.join(organized_dir, breed)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy(os.path.join(base_dir, img_name), os.path.join(class_dir, img_name))

print("✅ Dataset organized by breed.")

# Step 2: Prepare Dataset
img_size = (160, 160)
batch_size = 32

train_ds = image_dataset_from_directory(
    organized_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    organized_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Found classes:", class_names)

# Prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Step 3: Build Model with Transfer Learning
base_model = tf.keras.applications.MobileNetV2(input_shape=img_size + (3,),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train
history = model.fit(train_ds, validation_data=val_ds, epochs=5)

# Step 5: Save model and class names
model.save("oxford_pet_model.h5")
with open("class_names.txt", "w") as f:
    for c in class_names:
        f.write(c + "\n")

# Step 6: Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
