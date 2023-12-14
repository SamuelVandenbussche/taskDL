import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

NUM_CLASSES = 5
IMG_SIZE = 128
HEIGTH_FACTOR = 0.2
WIDTH_FACTOR = 0.2

model = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255),
  layers.RandomFlip("horizontal"),
  layers.RandomTranslation(HEIGTH_FACTOR,WIDTH_FACTOR),
  layers.RandomZoom(0.2),

  layers.Conv2D(32, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 3), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Conv2D(32, (3, 3), activation="relu"),
  # layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation="relu"),
  layers.Dense(128, activation="relu"),
  layers.Dropout(0.5),
  layers.Dense(128, activation="relu"),
  layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
batch_size = 32
image_size = (IMG_SIZE, IMG_SIZE)
validation_split = 0.2

train_set = image_dataset_from_directory(
    directory=r'C:\task_images\train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='training',
    seed=123
)

validation_set = image_dataset_from_directory(
    directory=r'C:\task_images\train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='validation',
    seed=123
)

# Create the testing dataset from the 'test' directory
test_set = image_dataset_from_directory(
    directory=r'C:\task_images\test',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size
)

# Streamlit app
st.title("TensorFlow Keras Model Training")

if st.button("Train Model"):
    history = model.fit(train_set, validation_data=validation_set, epochs=10)
    st.write("Training Accuracy: ", history.history['accuracy'][-1])
    st.write("Validation Accuracy: ", history.history['val_accuracy'][-1])
    st.write("Training Loss: ", history.history['loss'][-1])
    st.write("Validation Loss: ", history.history['val_loss'][-1])
    st.write("Model has been trained successfully!")
