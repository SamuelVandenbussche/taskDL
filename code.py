import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import image
from keras.utils import image_dataset_from_directory
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt


# Define the categories
categories = ["basketbal","golf bal","rugby bal","voetbal","tennis bal"]


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
st.image(row_images, caption=["basketbal","golf bal","rugby bal","voetbal","tennis bal"], width=150)

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

model = keras.Sequential([
   layers.Resizing(IMG_SIZE, IMG_SIZE),
   layers.Rescaling(1./255),
   layers.experimental.preprocessing.RandomFlip("horizontal"),
   layers.experimental.preprocessing.RandomTranslation(HEIGTH_FACTOR, WIDTH_FACTOR),
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

model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=[Accuracy()])

# Load and preprocess the data
batch_size = 32
image_size = (IMG_SIZE, IMG_SIZE)
validation_split = 0.2

train_set = image_dataset_from_directory(
    directory=r'.\task_images\train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='training',
    seed=123
)

validation_set = image_dataset_from_directory(
    directory=r'.\task_images\train',
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



st.header("Model Testing")

if uploaded_file is not None:

   # Preprocess the uploaded image
   img = Image.open(uploaded_file)
   img = img.resize((IMG_SIZE, IMG_SIZE))
   img_array = img_to_array(img)
   img_array = np.expand_dims(img_array, axis=0)
   img_array /= 255.0

   # Make predictions
   predictions = model.predict(img_array)

   # Display predictions
   st.write("Predictions:")
   for i, pred in enumerate(predictions[0]):
       st.write(f"{categories[i]}: {pred * 100:.2f}%")
