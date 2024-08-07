import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model("final_dia_CNN.h5")

# Streamlit app
st.title('Diabetes Prediction using Deep Learning')

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_file_path = "temp_image.png"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess the image
    img = load_img(temp_file_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction, axis=1)[0]

    diagnosis_dict_binary = {
        1: 'No_Diabetes (No_DR)',
        0: 'Diabetes (DR)'
    }

    result = diagnosis_dict_binary[pred_class]

    # Display the image and prediction result
    st.image(temp_file_path, caption='Uploaded Image', use_column_width=True)
    st.write("Prediction: ", result)

    # Clean up temporary file
    os.remove(temp_file_path)

# Plotting function for confusion matrix
def plot_confusion_matrix(con_mat):
    plt.figure(figsize=(10,10))
    sns.heatmap(con_mat, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='d')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    st.pyplot(plt)

# If you want to visualize the confusion matrix, you can add a button to generate it
#if st.button('Show Confusion Matrix'):
    # Assuming you have true labels and predictions stored in variables `y_true` and `y_pred`
   # y_true = [...]  # Replace with actual true labels
    #y_pred = [...]  # Replace with actual predicted labels
    
    #con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
    #plot_confusion_matrix(con_mat)
