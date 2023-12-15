import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Set Streamlit app title
st.title("AI Task Deep Learning")
st.write("This application is a deep learning model for classifying an image as a basketball, golf ball, rugby ball, soccer ball, or tennis ball.")

st.header("EDA")
# Display images
st.image("EDA.JPG", width=700)
# Display images
st.image(["1d50dc8500.jpg", "3a3a3b38f4.jpg","1b71d0b173.jpg", "3cda94891a.jpg","1a5c3f9b51.jpg"], width=140)

# Load data and train the model when the app is loaded
if "train_set" not in st.session_state:
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

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    st.session_state.train_set = train_set
    st.session_state.validation_set = validation_set
    st.session_state.model = model

# User input for epochs
epochs = st.number_input("Enter the number of epochs", min_value=1, value=10, step=1)

# Button to trigger model training
if st.button('Train the model'):
    # Model Training
    history = st.session_state.model.fit(
        st.session_state.train_set,
        validation_data=st.session_state.validation_set,
        epochs=epochs
    )

    st.session_state.history = history

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
    st.pyplot(fig)
st.header("only upload image when model is trained")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Check if an image has been uploaded
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", width=300)

    # Convert the uploaded image to a format compatible with the model
    img = Image.open(uploaded_file).resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    # Make predictions
    predictions = st.session_state.model.predict(img_array)
    class_labels = st.session_state.train_set.class_names
    predicted_category = class_labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    # Display predictions
    st.subheader("Prediction:")
    st.write(f"The uploaded image is most likely a '{predicted_category}' with {confidence:.2f}% confidence.")
