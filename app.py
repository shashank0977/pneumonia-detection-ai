import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import time

# Page config
st.set_page_config(page_title="Pneumonia Detector", page_icon="🫁", layout="wide")

# Load the trained model
@st.cache_resource
def load_pneumonia_model():
    return load_model("pneumonia_model.h5")

model = load_pneumonia_model()

# Sidebar navigation
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to", ["Home", "About"])

# ===============================
# 🏠 HOME PAGE
# ===============================
if page == "Home":
    st.title("🫁 Pneumonia Detection from Chest X-ray")
    st.markdown(
        """
        Welcome to the **AI-powered Pneumonia Detection System**!  
        Upload a **chest X-ray image**, and our deep learning model will predict whether
        pneumonia is detected.
        """
    )

    uploaded_file = st.file_uploader("📤 Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🩻 Uploaded X-ray", use_container_width=True)

        # Convert image to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Add small delay for animation
        with st.spinner("🔍 Analyzing the X-ray... Please wait"):
            time.sleep(2)
            prediction = model.predict(img_array)
            label = "🧠 Pneumonia Detected" if prediction[0][0] > 0.5 else "✅ Normal"
            confidence = prediction[0][0] if label.startswith("🧠") else 1 - prediction[0][0]

        # Animated result section
        st.success("✅ Analysis Complete!")
        st.markdown(f"## {label}")
        st.progress(float(confidence))

        st.markdown(
            f"### 📊 Confidence Level: **{confidence * 100:.2f}%**"
        )

        if label.startswith("🧠"):
            st.warning(
                """
                ⚠️ The X-ray shows signs consistent with pneumonia.
                Please consult a healthcare professional for a confirmed diagnosis.
                """
            )
        else:
            st.info("🎉 The X-ray appears normal. Keep maintaining good respiratory health!")

    else:
        st.info("👆 Please upload an image to start the analysis.")

# ===============================
# ℹ️ ABOUT PAGE
# ===============================
elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        ### 🧬 Overview
        This web app uses a **Convolutional Neural Network (CNN)** trained on chest X-ray images
        to detect the presence of pneumonia.

        ### ⚙️ How It Works
        1. You upload a chest X-ray image.
        2. The image is preprocessed and fed into a deep learning model.
        3. The model outputs a prediction with confidence level.

        ### 🧠 Technology Stack
        - **TensorFlow / Keras** for model training  
        - **Streamlit** for the web interface  
        - **NumPy & PIL** for image processing  

        ---
        👨‍⚕️ **Disclaimer:** This tool is for educational and research purposes only.  
        It should not replace professional medical advice or diagnosis.
        """
    )
