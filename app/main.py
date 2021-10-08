import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the dataset
data, metadata = tfds.load("cats_vs_dogs", as_supervised=True, with_info=True)

# Add the data to the train and test sets
train_data = {"images": [], "labels": []}
for i, (image, label) in enumerate(data["train"].take(25)):
    image = cv2.resize(image.numpy(), (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.reshae(100, 100, 1)

    train_data["images"].append(image)
    train_data["labels"].append(label)

train_data["images"] = np.array(train_data["images"]).astype(float) / 255
train_data["labels"] = np.array(train_data["labels"])

train_images = train_data["images"][:19700]
train_labels = train_data["labels"][:19700]

test_images = train_data["images"][19700:]
test_labels = train_data["labels"][19700:]

# Create the model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(100, 100, 1)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", input_shape=(100, 100, 1)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", input_shape=(100, 100, 1)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True,
)

datagen.fit(train_data["images"])

train_datagen = datagen.flow(train_images, train_labels, batch_size=32)

# Train the model
model.fit(
    train_datagen,
    batch_size=32,
    epochs=100,
    validation_data=(test_images, test_labels),
    steps_per_epoch=len(train_images) // 32,
    validation_steps=len(test_images) // 32
)

# Save the model
model.save("cats_vs_dogs_model.h5")

# Export the model
# mkdir output
# tensorflowjs_converter --input_format keras --output_dir output/ cats_vs_dogs_model.h5
