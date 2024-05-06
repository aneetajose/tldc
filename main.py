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
    st.write("The predicted disease is ", prediction)
    st.write("Solution:")
    if prediction == "Bacterial Spot":
        st.write("* Cultural Practices: Implement good cultural practices such as crop rotation, proper spacing between"
                 "plants, and avoiding overhead irrigation to minimize the spread of bacteria.\n\n"
                 "* Sanitation: Remove and destroy infected plant debris to reduce the source of bacteria.\n\n"
                 "* Copper-based Sprays: Copper-based fungicides or bactericides can be applied preventatively or at "
                 "the first sign of disease to help control bacterial spot.\n\n"
                 "* Biological Control: Some beneficial bacteria and fungi can compete with the pathogenic bacteria "
                 "responsible for bacterial spot. Biological control agents can be applied to the plants to suppress "
                 "the disease.\n\n"
                 "* Resistant Varieties: Planting tomato varieties that are resistant to bacterial spot can help reduce"
                 "the impact of the disease.")
    else:
        st.write("* Vector Control: Yellow leaf curl virus is often transmitted by whiteflies. Implement measures to "
                 "control whitefly populations, such as using sticky traps, applying insecticidal soaps or oils, "
                 "or introducing natural predators of whiteflies.\n\n"
                 "* Resistant Varieties: Some tomato varieties have been bred to be resistant to yellow leaf curl "
                 "virus. Planting these varieties can help reduce the incidence of the disease.\n\n"
                 "* Sanitation: Remove and destroy infected plants to prevent the spread of the virus to healthy "
                 "plants."
                 "* Reflective Mulches: Reflective mulches can deter whiteflies, reducing the transmission of the virus"
                 "to tomato plants.\n\n"
                 "* Avoid Over-fertilization: Excessive nitrogen fertilization can increase susceptibility to yellow "
                 "leaf curl virus. Use fertilizers judiciously to avoid promoting lush growth that is attractive to "
                 "whiteflies.")
