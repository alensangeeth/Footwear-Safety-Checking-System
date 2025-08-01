import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# Image dimensions and paths
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
DATASET_DIR = 'footwear_dataset'

# Data augmentation and split

# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,  # 80% train, 20% validation
#     rotation_range=15,
#     zoom_range=0.2,
#     shear_range=0.2,
#     horizontal_flip=True
# )

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False,
    fill_mode='nearest'
)

# Train generator
train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation generator
val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# CNN Model
model = Sequential([
    # Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    # Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # For 3-class classification
])

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights_dict = dict(enumerate(class_weights))


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# Train the model
# model.fit(train_data, validation_data=val_data, epochs=25)
model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    class_weight=class_weights_dict
)

# Save the model
os.makedirs('model', exist_ok=True)
model.save('model/footwear_safety_model.h5')
