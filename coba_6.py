import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import tensorflow as tf

st.set_page_config(page_title="🎀 Image App", layout="centered")

# ================= CSS Theme Pink 🎀 =================
st.markdown("""
<style>

body {
    background: #ffe6f2;
}

h1, h2, h3, label, .stButton>button {
    color: #b30059 !important;
}

.stButton>button {
    background-color: #ff99c8 !important;
    border-radius: 12px;
    border: 2px solid white;
}

.ribbon {
    position: fixed;
    top: -10vh;
    animation: fall 5s linear infinite;
    opacity: 0.9;
}

@keyframes fall {
    0% { transform: translateY(-20vh) rotate(0deg); }
    100% { transform: translateY(110vh) rotate(180deg); }
}
</style>
""", unsafe_allow_html=True)


# ========== Generate Falling Ribbons 🎀 ==========
import random

def falling_ribbons():
    ribbons = ""
    for _ in range(30):
        x = random.randint(0, 100)
        size = random.randint(15, 40)
        ribbons += f"""
        <span class="ribbon" style="left:{x}vw; font-size:{size}px;">🎀</span>
        """
    st.markdown(ribbons, unsafe_allow_html=True)

falling_ribbons()

st.markdown("<h1 style='text-align:center'>🎀 Image Classification & Object Detection App 🎀</h1>", unsafe_allow_html=True)


# ================= Load Model =================
@st.cache_resource
def load_model():
    yolo = YOLO("model/best.pt")
    classifier = tf.keras.models.load_model("model/best.h5")
    return yolo, classifier

yolo_model, classifier_model = load_model()


# Sidebar Mode Selection
mode = st.sidebar.radio("Mode:",
                        ["Object Detection (YOLO)", "Image Classification (.h5 Model)"],
                        index=0)


# ================= Upload Image =================
uploaded_file = st.file_uploader("Upload image...",
                                 type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image Uploaded ✅", use_column_width=True)

    if st.button("🎀 Process Image 🎀"):
        # Object Detection Mode
        if mode == "Object Detection (YOLO)":
            results = yolo_model.predict(np.array(img))
            result_img = results[0].plot()
            st.image(result_img, caption="🎯 Object Detection Result")
            st.success("Object detection done ✅")

        # Classification Mode
        else:
            img_resized = img.resize((224, 224))
            img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
            result = classifier_model.predict(img_array)
            class_idx = int(np.argmax(result))
            st.success(f"✅ Predicted Class: {class_idx}")
