import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
import re

st.set_page_config(page_title="Dog Breed Detector", layout="wide")

# ---------------------------
# LOAD MODEL + DATA
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_breed_resnet.keras")

model = load_model()

@st.cache_data
def load_labels():
    with open("class_indices.json") as f:
        data = json.load(f)
    return {int(v): k for k, v in data.items()}

label_map = load_labels()

@st.cache_data
def load_breed_info():
    with open("120_breeds_new.json") as f:
        data = json.load(f)
    return {item["Breed"]: item for item in data}

breed_info = load_breed_info()

# ---------------------------
# NORMALIZE FUNCTION
# ---------------------------
def normalize(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())

# ---------------------------
# BREED DETAILS
# ---------------------------
def get_breed_details(pred_breed):
    pred_clean = normalize(pred_breed)

    # Exact match
    for breed in breed_info:
        if normalize(breed) == pred_clean:
            return format_info(breed)

    # Partial match
    for breed in breed_info:
        if pred_clean in normalize(breed):
            return format_info(breed)

    return "No details available."

def format_info(breed):
    info = breed_info[breed]
    text = f"### 📘 About {breed}\n\n"
    for k, v in info.items():
        if k != "Breed":
            text += f"**{k}:** {v}\n\n"
    return text

# ---------------------------
# PREDICT BREED
# ---------------------------
def predict_breed(img):
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)

    pred = model.predict(arr)
    idx = int(np.argmax(pred))
    breed = label_map[idx]
    conf = float(np.max(pred) * 100)
    return breed, conf

# ---------------------------
# HISTORY
# ---------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
page = st.sidebar.radio("Navigate", ["🏠 Home", "🐶 Breed Detector", "📜 Prediction History"])

# ---------------------------
# HOME PAGE
# ---------------------------
if page == "🏠 Home":
    st.title("🐾 Dog Breed Detection System")
    st.write("""
    - Detect breed of any dog  
    - View detailed breed information  
    - Check prediction history  
    - Use AI Chatbot for dog-related questions  
    """)

    st.write("👉 **Chatbot is available in sidebar under `Chatbot` tab**")

# ---------------------------
# DETECTOR PAGE
# ---------------------------
elif page == "🐶 Breed Detector":
    st.title("🐶 Dog Breed Detector")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=300)

        breed, conf = predict_breed(img)

        st.success(f"### 🐾 Breed: **{breed}**")
        st.info(f"### Confidence: **{conf:.2f}%**")

        st.session_state.history.append({
            "image": img,
            "breed": breed,
            "conf": conf,
        })

        if st.button("Know More 🐾"):
            st.markdown(get_breed_details(breed))

# ---------------------------
# HISTORY PAGE
# ---------------------------
elif page == "📜 Prediction History":
    st.title("📜 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet.")
    else:
        for h in st.session_state.history:
            st.image(h["image"], width=200)
            st.write(f"Breed: **{h['breed']}**")
            st.write(f"Confidence: **{h['conf']:.2f}%**")
            st.markdown("---")
