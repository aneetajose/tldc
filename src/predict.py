import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array


# Load the model
model = load_model('../model/classifier.keras')
labels = [Bacterial_Spot, Yellow_Curl_Leaf]
# Load the test image
img_path = r'..\Dataset\test_data'
img = load_img(img_path, target_size=(256, 256))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

plt.imshow(img)
plt.axis('off')
plt.show()

# Make prediction
predictions = model.predict(img_array)
predicted_class = labels[np.argmax(predictions)]
print(predicted_class)
