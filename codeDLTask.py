import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# Define the categories
categories = ["basketbal", "golf bal", "rugby bal", "voetbal", "tennis bal"]

# Set Streamlit app title
st.title("AI Task Deep Learning")

# User input for epochs
epochs = st.number_input("Enter the number of epochs", min_value=1, value=10, step=1)

data_loaded = False

# Model definition
NUM_CLASSES = 5
IMG_SIZE = 128
batch_size = 32
image_size = (IMG_SIZE, IMG_SIZE)
validation_split = 0.2

train_set = image_dataset_from_directory(
    directory=r'task_images/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='training',
    seed=123
)
validation_set = image_dataset_from_directory(
    directory=r'task_images/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='validation',
    seed=123
)

# Model definition
model = keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255),
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomTranslation(0.2, 0.2),
    layers.experimental.preprocessing.RandomZoom(0.2),
    layers.Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

if st.button('Train the model'):

    # Model Training
    history = model.fit(
        train_set,
        validation_data=validation_set,
        epochs=epochs
    )

    # Display training metrics
    st.header("Training Metrics")

    # Plot loss and accuracy curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(history.history['loss'], label='training loss')
    ax1.plot(history.history['val_loss'], label='validation loss')
    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='training accuracy')
    ax2.plot(history.history['val_accuracy'], label='validation accuracy')
    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    fig.tight_layout()

    # Show the plots
    st.pyplot(fig)
