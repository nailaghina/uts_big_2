import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # Preprocessing
        img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        st.write("### Hasil Prediksi:", class_index)
        st.write("Probabilitas:", np.max(prediction))

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import cv2

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


