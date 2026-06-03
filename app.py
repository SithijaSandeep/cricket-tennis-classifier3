import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Ball Classifier",
    page_icon="🏏",
    layout="centered"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.main {
    background: linear-gradient(to bottom right, #0f172a, #1e293b);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}

.stFileUploader {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
}

.result-box {
    background: linear-gradient(to right, #2563eb, #7c3aed);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-top: 20px;
}

.prediction-text {
    font-size: 30px;
    font-weight: bold;
}

.confidence-text {
    font-size: 20px;
    margin-top: 10px;
}

.image-box {
    border-radius: 20px;
    overflow: hidden;
    border: 3px solid #334155;
    margin-top: 20px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model("ball_classifier.keras")

# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ============================================================
# IMAGE SIZE
# ============================================================

IMG_SIZE = (160, 160)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🏏 Cricket Ball vs Tennis Ball Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload an image and let AI identify the ball type</div>',
    unsafe_allow_html=True
)

# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Ball Image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.markdown('<div class="image-box">', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

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

    # Spinner
    with st.spinner("Analyzing Image..."):

        # Predict
        prediction = model.predict(img_array)

    # Get prediction
    predicted_class = class_names[np.argmax(prediction)]

    # Confidence
    confidence = np.max(prediction) * 100

    # ========================================================
    # RESULT UI
    # ========================================================

    st.markdown(f"""
    <div class="result-box">

        <div class="prediction-text">
            Prediction: {predicted_class}
        </div>

        <div class="confidence-text">
            Confidence: {confidence:.2f}%
        </div>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align:center; color:#94a3b8; font-size:14px;'>
        Powered by TensorFlow & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import json

# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # Load model
# model = tf.keras.models.load_model("ball_classifier.keras")

# # Load class names
# with open("class_names.json", "r") as f:
#     class_names = json.load(f)

# # Image size
# IMG_SIZE = (160, 160)

# # Title
# st.title("Cricket Ball vs Tennis Ball Classifier")

# st.write("Upload an image to identify the ball type")

# # Upload image
# uploaded_file = st.file_uploader(
#     "Upload Ball Image",
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file is not None:

#     # Open image
#     image = Image.open(uploaded_file).convert("RGB")

#     # Show image
#     st.image(image, caption="Uploaded Image")

#     # Resize image
#     image = image.resize(IMG_SIZE)

#     # Convert to numpy
#     img_array = np.array(image)

#     # Convert to float32
#     img_array = img_array.astype(np.float32)

#     # Preprocess image
#     img_array = preprocess_input(img_array)

#     # Expand dimensions
#     img_array = np.expand_dims(img_array, axis=0)

#     # Predict
#     prediction = model.predict(img_array)

#     # Get result
#     predicted_class = class_names[np.argmax(prediction)]

#     # Confidence
#     confidence = np.max(prediction) * 100

#     # Show output
#     st.success(f"Prediction: {predicted_class}")




    
#     st.info(f"Confidence: {confidence:.2f}%")

