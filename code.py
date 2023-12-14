import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Resizing, Rescaling, RandomFlip, RandomTranslation, RandomZoom, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Set Streamlit app title
st.title("AI Task Deep Learning")

# Add EDA section
st.header("Exploratory Data Analysis (EDA)")

# Add a sample picture
sample_image_path = "./EDA.JPG"  # Replace with the path to your sample image
sample_image = Image.open(sample_image_path)
st.image(sample_image, caption="Sample Picture", use_column_width=True)

# List of paths to five images (replace with actual paths)
image_paths = ["./1a5c3f9b51.jpg", "./1b71d0b173.jpg", "./1d50dc8500.jpg", "./3a3a3b38f4.jpg", "./3cda94891a.jpg"]

# Display images in a row
row_images = [Image.open(image_path) for image_path in image_paths]
st.image(row_images, caption=["tennis bal", "voetbal", "rugby bal", "golf bal", "basketbal"], width=150)

# Add section for uploading a picture
st.header("Upload a Picture")

# Upload picture button
uploaded_file = st.file_uploader("Choose a picture...", type="jpg")

# Display the uploaded picture
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Picture", use_column_width=True)

# Model Training Section
st.header("Model Training")

# User input for epochs
epochs = st.number_input("Enter the number of epochs", min_value=1, value=10, step=1)

# Model definition
NUM_CLASSES = 5
IMG_SIZE = 128
HEIGTH_FACTOR = 0.2
WIDTH_FACTOR = 0.2

model = Sequential([
    Resizing(IMG_SIZE, IMG_SIZE),
    Rescaling(1./255),
    RandomFlip("horizontal"),
    RandomTranslation(HEIGTH_FACTOR, WIDTH_FACTOR),
    RandomZoom(0.2),
    Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(32, (3, 3), activation="relu"),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=[Accuracy()])

# Load and preprocess the data
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

# Save the model
model.save("trained_model.tf")
st.success("Model trained and saved successfully!")

# Testing Section
st.header("Model Testing")

if uploaded_file is not None:
    # Load the model
    loaded_model = tf.keras.models.load_model("trained_model.tf")

    # Preprocess the uploaded image
    img = Image.open(uploaded_file)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = loaded_model.predict(img_array)

    # Display predictions
    st.write("Predictions:")
    for i, pred in enumerate(predictions[0]):
        st.write(f"{categories[i]}: {pred * 100:.2f}%")
