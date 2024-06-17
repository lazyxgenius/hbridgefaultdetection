import joblib  # to save trained model..for serialization and deserialization of Python objects.
import setuptools  # for tensor flow (distutils)
import tensorflow as tf
import keras
import warnings

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout   # layers in NN
from tensorflow.keras.models import Sequential  # model. linear stack of layers
from tensorflow.keras.layers import Activation  # for activation functions
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam  # optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adamax

warnings.filterwarnings("ignore")

num_classes = 9

# Create a Sequential model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))  # Convolutional layer with 32 filters and 3x3 kernel
model.add(MaxPooling2D((2, 2)))  # Max-pooling layer with 2x2 pool size.
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the output and add fully connected layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))  # Fully connected layer with 256 units and ReLU activation.
model.add(Dropout(0.5))  # Dropout layer with a rate of 0.5.
model.add(Dense(num_classes, activation='softmax'))  # Output layer with num_classes units and softmax activation.

# Compile the model
opt = Adamax(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# Used ADAM, ADAMAX, SGD optimizers. Best results - ADAMAX (learning rate = 0.001, batch size 16)
# results to all are in accuracies.png
# graph of the best result is in ADAMAX-lr=0.001=BS=16.png
# detailed report is report1.pdf

model.summary()

# paths to dataset
train_data_dir = r"C:\Users\Aditya PC\PycharmProjects\hbridgefaultdetection\New dataset\Train"
validation_data_dir = r"C:\Users\Aditya PC\PycharmProjects\hbridgefaultdetection\New dataset\Validation"
# this is a small dataset(100 images per class), in practice we used over 1000 images per class.


# Set image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 16

train_datagen = ImageDataGenerator(     # for training with various augmentations.LEARN WHAT ARE THESE AUGMENTATIONS
    rescale=1.0 / 225,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 225,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess the data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)


model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=100
)

filename = "saved_hbridge.joblib"
joblib.dump(model, filename)

# Plot accuracy and loss versus epochs
plt.figure(figsize=(12, 4))

# Accessing history from the History object
history = model.history.history

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


