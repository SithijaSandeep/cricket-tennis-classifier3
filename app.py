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
    page_title="Neon AI Ball Classifier",
    page_icon="⚡",
    layout="centered"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Google Font */

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background */

.stApp {

    background: linear-gradient(
        -45deg,
        #0f0c29,
        #302b63,
        #24243e,
        #1f1147,
        #3a1c71
    );

    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}

/* Animated Gradient */

@keyframes gradient {

    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }
}

/* Main Glass Container */

.main-container {

    background: rgba(255,255,255,0.08);

    backdrop-filter: blur(18px);

    border: 1px solid rgba(255,255,255,0.15);

    border-radius: 30px;

    padding: 40px;

    box-shadow:
        0px 8px 32px rgba(0,0,0,0.45),
        0px 0px 20px rgba(168,85,247,0.2);

}

/* Main Title */

.main-title {

    text-align: center;

    font-size: 52px;

    font-weight: 800;

    background: linear-gradient(
        to right,
        #38bdf8,
        #818cf8,
        #c084fc,
        #f472b6
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    margin-bottom: 10px;
}

/* Subtitle */

.sub-title {

    text-align: center;

    color: #dbeafe;

    font-size: 18px;

    margin-bottom: 35px;
}

/* Upload Box */

[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.06);

    border: 2px dashed #a855f7;

    border-radius: 22px;

    padding: 25px;

    transition: 0.3s ease;
}

[data-testid="stFileUploader"]:hover {

    border-color: #38bdf8;

    transform: scale(1.01);
}

/* Uploaded Image */

img {

    border-radius: 22px;

    border: 3px solid rgba(255,255,255,0.15);

    box-shadow:
        0px 10px 30px rgba(0,0,0,0.5),
        0px 0px 20px rgba(56,189,248,0.25);
}

/* Result Card */

.result-card {

    margin-top: 30px;

    background: linear-gradient(
        135deg,
        #06b6d4,
        #3b82f6,
        #8b5cf6,
        #ec4899
    );

    padding: 30px;

    border-radius: 28px;

    text-align: center;

    color: white;

    box-shadow:
        0px 10px 35px rgba(139,92,246,0.5),
        0px 0px 25px rgba(236,72,153,0.3);
}

/* Prediction Text */

.prediction {

    font-size: 34px;

    font-weight: 700;

    margin-bottom: 15px;
}

/* Confidence */

.confidence {

    font-size: 22px;

    font-weight: 500;
}

/* Footer */

.footer {

    text-align: center;

    color: #cbd5e1;

    margin-top: 35px;

    font-size: 14px;
}

/* Spinner Text */

.stSpinner > div {

    color: white !important;
}

/* Hide Streamlit Branding */

footer {
    visibility: hidden;
}

header {
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
# MAIN UI
# ============================================================

st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">⚡ AI Ball Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Modern Deep Learning Image Classification Experience</div>',
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
# PREDICTION SECTION
# ============================================================

if uploaded_file is not None:

    # Open Image
    image = Image.open(uploaded_file).convert("RGB")

    # Display Image
    st.image(
        image,
        use_container_width=True
    )

    # Resize Image
    image = image.resize(IMG_SIZE)

    # Convert to Array
    img_array = np.array(image)

    # Float32
    img_array = img_array.astype(np.float32)

    # Preprocess
    img_array = preprocess_input(img_array)

    # Expand Dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Loading Animation
    with st.spinner("⚡ AI is analyzing your image..."):

        prediction = model.predict(img_array)

    # Get Prediction
    predicted_class = class_names[np.argmax(prediction)]

    # Confidence
    confidence = np.max(prediction) * 100

    # ========================================================
    # RESULT CARD
    # ========================================================

    st.markdown(f"""
    <div class="result-card">

        <div class="prediction">
            🎯 Prediction: {predicted_class}
        </div>

        <div class="confidence">
            📊 Confidence: {confidence:.2f}%
        </div>

    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🚀 Powered by TensorFlow • Streamlit • MobileNetV2
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import json

# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # ============================================================
# # PAGE CONFIG
# # ============================================================

# st.set_page_config(
#     page_title="AI Ball Classifier",
#     page_icon="🎯",
#     layout="centered"
# )

# # ============================================================
# # CUSTOM CSS
# # ============================================================

# st.markdown("""
# <style>

# /* Main Background */

# .stApp {
#     background: linear-gradient(
#         135deg,
#         #0f172a,
#         #111827,
#         #1e1b4b,
#         #312e81
#     );
#     background-size: 400% 400%;
#     animation: gradientBG 15s ease infinite;
# }

# /* Animated Background */

# @keyframes gradientBG {
#     0% {
#         background-position: 0% 50%;
#     }

#     50% {
#         background-position: 100% 50%;
#     }

#     100% {
#         background-position: 0% 50%;
#     }
# }

# /* Main Container */

# .main-box {
#     background: rgba(255,255,255,0.08);
#     backdrop-filter: blur(15px);
#     border: 1px solid rgba(255,255,255,0.15);
#     border-radius: 25px;
#     padding: 35px;
#     box-shadow: 0px 10px 40px rgba(0,0,0,0.4);
# }

# /* Title */

# .main-title {
#     text-align: center;
#     font-size: 48px;
#     font-weight: 800;
#     color: white;
#     margin-bottom: 10px;
# }

# /* Subtitle */

# .sub-title {
#     text-align: center;
#     font-size: 18px;
#     color: #dbeafe;
#     margin-bottom: 35px;
# }

# /* Upload Box */

# [data-testid="stFileUploader"] {
#     background: rgba(255,255,255,0.08);
#     border: 2px dashed #60a5fa;
#     border-radius: 20px;
#     padding: 25px;
# }

# /* Uploaded Image */

# .image-container img {
#     border-radius: 20px;
#     border: 4px solid rgba(255,255,255,0.2);
#     box-shadow: 0px 8px 25px rgba(0,0,0,0.5);
# }

# /* Result Card */

# .result-card {
#     background: linear-gradient(
#         135deg,
#         #06b6d4,
#         #3b82f6,
#         #8b5cf6
#     );

#     padding: 25px;
#     border-radius: 25px;
#     text-align: center;
#     color: white;
#     margin-top: 30px;

#     box-shadow: 0px 10px 30px rgba(59,130,246,0.4);
# }

# /* Prediction Text */

# .prediction-text {
#     font-size: 34px;
#     font-weight: bold;
#     margin-bottom: 12px;
# }

# /* Confidence Text */

# .confidence-text {
#     font-size: 22px;
# }

# /* Footer */

# .footer {
#     text-align: center;
#     color: #cbd5e1;
#     margin-top: 35px;
#     font-size: 14px;
# }

# /* Hide Streamlit Footer */

# footer {
#     visibility: hidden;
# }

# header {
#     visibility: hidden;
# }

# </style>
# """, unsafe_allow_html=True)

# # ============================================================
# # LOAD MODEL
# # ============================================================

# model = tf.keras.models.load_model("ball_classifier.keras")

# # ============================================================
# # LOAD CLASS NAMES
# # ============================================================

# with open("class_names.json", "r") as f:
#     class_names = json.load(f)

# # ============================================================
# # IMAGE SIZE
# # ============================================================

# IMG_SIZE = (160, 160)

# # ============================================================
# # MAIN UI
# # ============================================================

# st.markdown('<div class="main-box">', unsafe_allow_html=True)

# st.markdown(
#     '<div class="main-title">🎯 AI Ball Classifier</div>',
#     unsafe_allow_html=True
# )

# st.markdown(
#     '<div class="sub-title">Upload a Cricket Ball or Tennis Ball image and let AI identify it instantly</div>',
#     unsafe_allow_html=True
# )

# # ============================================================
# # FILE UPLOADER
# # ============================================================

# uploaded_file = st.file_uploader(
#     "Upload Ball Image",
#     type=["jpg", "jpeg", "png"]
# )

# # ============================================================
# # PREDICTION SECTION
# # ============================================================

# if uploaded_file is not None:

#     # Open Image
#     image = Image.open(uploaded_file).convert("RGB")

#     # Show Image
#     st.markdown('<div class="image-container">', unsafe_allow_html=True)

#     st.image(
#         image,
#         use_container_width=True
#     )

#     st.markdown('</div>', unsafe_allow_html=True)

#     # Resize Image
#     image = image.resize(IMG_SIZE)

#     # Convert to Array
#     img_array = np.array(image)

#     # Convert to float32
#     img_array = img_array.astype(np.float32)

#     # Preprocess
#     img_array = preprocess_input(img_array)

#     # Expand Dimensions
#     img_array = np.expand_dims(img_array, axis=0)

#     # Loading Animation
#     with st.spinner("🤖 AI is analyzing the image..."):

#         # Prediction
#         prediction = model.predict(img_array)

#     # Get Prediction
#     predicted_class = class_names[np.argmax(prediction)]

#     # Confidence
#     confidence = np.max(prediction) * 100

#     # ========================================================
#     # RESULT CARD
#     # ========================================================

#     st.markdown(f"""
#     <div class="result-card">

#         <div class="prediction-text">
#             🧠 Prediction: {predicted_class}
#         </div>

#         <div class="confidence-text">
#             📊 Confidence: {confidence:.2f}%
#         </div>

#     </div>
#     """, unsafe_allow_html=True)

# # ============================================================
# # FOOTER
# # ============================================================

# st.markdown(
#     """
#     <div class="footer">
#         🚀 Powered by TensorFlow + Streamlit
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# st.markdown('</div>', unsafe_allow_html=True)

# # import streamlit as st
# # import tensorflow as tf
# # import numpy as np
# # from PIL import Image
# # import json

# # from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # # ============================================================
# # # PAGE CONFIG
# # # ============================================================

# # st.set_page_config(
# #     page_title="Ball Classifier",
# #     page_icon="🏏",
# #     layout="centered"
# # )

# # # ============================================================
# # # CUSTOM CSS
# # # ============================================================

# # st.markdown("""
# # <style>

# # body {
# #     background-color: #0f172a;
# # }

# # .main {
# #     background: linear-gradient(to bottom right, #0f172a, #1e293b);
# #     color: white;
# # }

# # .block-container {
# #     padding-top: 2rem;
# # }

# # .title {
# #     text-align: center;
# #     font-size: 42px;
# #     font-weight: bold;
# #     color: white;
# #     margin-bottom: 5px;
# # }

# # .subtitle {
# #     text-align: center;
# #     color: #cbd5e1;
# #     font-size: 18px;
# #     margin-bottom: 30px;
# # }

# # .stFileUploader {
# #     background-color: #1e293b;
# #     padding: 20px;
# #     border-radius: 15px;
# #     border: 1px solid #334155;
# # }

# # .result-box {
# #     background: linear-gradient(to right, #2563eb, #7c3aed);
# #     padding: 20px;
# #     border-radius: 15px;
# #     text-align: center;
# #     color: white;
# #     margin-top: 20px;
# # }

# # .prediction-text {
# #     font-size: 30px;
# #     font-weight: bold;
# # }

# # .confidence-text {
# #     font-size: 20px;
# #     margin-top: 10px;
# # }

# # .image-box {
# #     border-radius: 20px;
# #     overflow: hidden;
# #     border: 3px solid #334155;
# #     margin-top: 20px;
# # }

# # footer {
# #     visibility: hidden;
# # }

# # </style>
# # """, unsafe_allow_html=True)

# # # ============================================================
# # # LOAD MODEL
# # # ============================================================

# # model = tf.keras.models.load_model("ball_classifier.keras")

# # # ============================================================
# # # LOAD CLASS NAMES
# # # ============================================================

# # with open("class_names.json", "r") as f:
# #     class_names = json.load(f)

# # # ============================================================
# # # IMAGE SIZE
# # # ============================================================

# # IMG_SIZE = (160, 160)

# # # ============================================================
# # # HEADER
# # # ============================================================

# # st.markdown(
# #     '<div class="title">🏏 Cricket Ball vs Tennis Ball Classifier</div>',
# #     unsafe_allow_html=True
# # )

# # st.markdown(
# #     '<div class="subtitle">Upload an image and let AI identify the ball type</div>',
# #     unsafe_allow_html=True
# # )

# # # ============================================================
# # # FILE UPLOADER
# # # ============================================================

# # uploaded_file = st.file_uploader(
# #     "Upload Ball Image",
# #     type=["jpg", "jpeg", "png"]
# # )

# # # ============================================================
# # # PREDICTION
# # # ============================================================

# # if uploaded_file is not None:

# #     # Open image
# #     image = Image.open(uploaded_file).convert("RGB")

# #     # Display image
# #     st.markdown('<div class="image-box">', unsafe_allow_html=True)
# #     st.image(image, use_container_width=True)
# #     st.markdown('</div>', unsafe_allow_html=True)

# #     # Resize image
# #     image = image.resize(IMG_SIZE)

# #     # Convert to numpy
# #     img_array = np.array(image)

# #     # Convert to float32
# #     img_array = img_array.astype(np.float32)

# #     # Preprocess image
# #     img_array = preprocess_input(img_array)

# #     # Expand dimensions
# #     img_array = np.expand_dims(img_array, axis=0)

# #     # Spinner
# #     with st.spinner("Analyzing Image..."):

# #         # Predict
# #         prediction = model.predict(img_array)

# #     # Get prediction
# #     predicted_class = class_names[np.argmax(prediction)]

# #     # Confidence
# #     confidence = np.max(prediction) * 100

# #     # ========================================================
# #     # RESULT UI
# #     # ========================================================

# #     st.markdown(f"""
# #     <div class="result-box">

# #         <div class="prediction-text">
# #             Prediction: {predicted_class}
# #         </div>

# #         <div class="confidence-text">
# #             Confidence: {confidence:.2f}%
# #         </div>

# #     </div>
# #     """, unsafe_allow_html=True)

# # # ============================================================
# # # FOOTER
# # # ============================================================

# # st.markdown("<br><br>", unsafe_allow_html=True)

# # st.markdown(
# #     """
# #     <div style='text-align:center; color:#94a3b8; font-size:14px;'>
# #         Powered by TensorFlow & Streamlit
# #     </div>
# #     """,
# #     unsafe_allow_html=True
# # )
# # # import streamlit as st
# # # import tensorflow as tf
# # # import numpy as np
# # # from PIL import Image
# # # import json

# # # from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # # # Load model
# # # model = tf.keras.models.load_model("ball_classifier.keras")

# # # # Load class names
# # # with open("class_names.json", "r") as f:
# # #     class_names = json.load(f)

# # # # Image size
# # # IMG_SIZE = (160, 160)

# # # # Title
# # # st.title("Cricket Ball vs Tennis Ball Classifier")

# # # st.write("Upload an image to identify the ball type")

# # # # Upload image
# # # uploaded_file = st.file_uploader(
# # #     "Upload Ball Image",
# # #     type=["jpg", "jpeg", "png"]
# # # )

# # # if uploaded_file is not None:

# # #     # Open image
# # #     image = Image.open(uploaded_file).convert("RGB")

# # #     # Show image
# # #     st.image(image, caption="Uploaded Image")

# # #     # Resize image
# # #     image = image.resize(IMG_SIZE)

# # #     # Convert to numpy
# # #     img_array = np.array(image)

# # #     # Convert to float32
# # #     img_array = img_array.astype(np.float32)

# # #     # Preprocess image
# # #     img_array = preprocess_input(img_array)

# # #     # Expand dimensions
# # #     img_array = np.expand_dims(img_array, axis=0)

# # #     # Predict
# # #     prediction = model.predict(img_array)

# # #     # Get result
# # #     predicted_class = class_names[np.argmax(prediction)]

# # #     # Confidence
# # #     confidence = np.max(prediction) * 100

# # #     # Show output
# # #     st.success(f"Prediction: {predicted_class}")




    
# # #     st.info(f"Confidence: {confidence:.2f}%")

