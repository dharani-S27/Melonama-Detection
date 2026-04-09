import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import gdown
import os

if not os.path.exists("final_melanoma_model.h5"):
    url = "https://drive.google.com/file/d/1v9S5FCCGLgdotf28H6GCnBhU7nBFKnx8/view?usp=drive_link"
    gdown.download(url, "final_melanoma_model.h5", quiet=False)

model = load_model("/content/final_melanoma_model.h5")

st.title("Melanoma Detection App")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

age = st.slider("Age", 0, 100, 25)
sex = st.selectbox("Gender", ["Male","Female"])
sex = 1 if sex=="Male" else 0

def preprocess(image):
    image = image.resize((224,224))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    return image

if st.button("Predict"):
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img)

        img = preprocess(img)
        clinical = np.array([[age, sex]])

        pred = model.predict([img, clinical])
        prob = pred[0][0]

        if prob > 0.5:
            st.error(f"Malignant - {prob:.2f}")
        else:
            st.success(f"Benign - {1-prob:.2f}")
