
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
model = tf.keras.models.load_model("ball_classifier.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Image size
IMG_SIZE = (160, 160)

# Title
st.title("Cricket Ball vs Tennis Ball Classifier")

st.write("Upload an image to identify the ball type")

# Upload image
uploaded_file = st.file_uploader(
    "Upload Ball Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Show image
    st.image(image, caption="Uploaded Image")

    # Resize image
    image = image.resize(IMG_SIZE)

    # Convert to numpy
    img_array = np.array(image)

    # Convert to float32
    img_array = img_array.astype(np.float32)

    # Preprocess image
    img_array = preprocess_input(img_array)

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # Get result
    predicted_class = class_names[np.argmax(prediction)]

    # Confidence
    confidence = np.max(prediction) * 100

    # Show output
    st.success(f"Prediction: {predicted_class}")




    
    st.info(f"Confidence: {confidence:.2f}%")
```
