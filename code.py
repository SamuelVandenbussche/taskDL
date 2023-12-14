import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit app title
st.title("AI Task Deep Learning")

# Add EDA section
st.header("Exploratory Data Analysis (EDA)")


# Add section for uploading a picture
st.header("Upload a Picture")

# Upload picture button
uploaded_file = st.file_uploader("Choose a picture...", type="jpg")

# Display the uploaded picture
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Picture", use_column_width=True)

# Additional content can be added as needed
