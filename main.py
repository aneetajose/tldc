import streamlit as st
from PIL import Image
from src.predict import detect


st.title('Tomato Leaf Disease Detection')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image_data = Image.open(uploaded_file)
    prediction = detect(uploaded_file)
    st.image(image_data, caption='Uploaded Image')
    st.write("Prediction:")
    st.write(prediction)