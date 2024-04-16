
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from PIL import Image


def detect(img_path):
    # Load the model
    model = load_model('../model/classifier.keras')
    labels = ["Bacterial_Spot", "Yellow_Curl_Leaf"]

    img = Image.open(img_path)
    img_resized = img.resize((256, 256))
    img_array = np.array(img_resized)
    img_array_normalized = img_array / 255.0

    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = labels[np.argmax(predictions)]
    return predicted_class


if __name__ == '__main__':
    imgpath = r'..\Dataset\test_data\Tomato___Bacterial_spot\0ab41c2e-c6fc-4ef1-9ffb-ce1b241d32be___GCREC_Bact.Sp 3426.JPG'
    detect(imgpath)
