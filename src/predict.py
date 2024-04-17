import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from PIL import Image


def detect(img_path):
    # Load the model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '..', 'model')
    model_path = os.path.join(model_dir, 'classifier.keras')
    model = load_model(model_path)
    labels = ["Bacterial_Spot", "Yellow_Curl_Leaf"]

    img = Image.open(img_path)
    img_resized = img.resize((256, 256))
    img_array = np.array(img_resized)
    img_array_normalized = img_array / 255.0
    img_array_normalized = np.expand_dims(img_array_normalized, axis=0)  # Add batch dimension

    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

    # Make prediction
    predictions = model.predict(img_array_normalized)
    predicted_class = labels[np.argmax(predictions)]
    return predicted_class


if __name__ == '__main__':
    imgpath = r'..\test-images\test_1.JPG'
    print(detect(imgpath))
