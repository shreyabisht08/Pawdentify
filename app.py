import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import google.generativeai as genai
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
import re

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Dog Breed Detector", layout="wide")

# ------------------------------------------------------
# DARK / LIGHT MODE
# ------------------------------------------------------
if "dark" not in st.session_state:
    st.session_state.dark = False

def switch_theme():
    st.session_state.dark = not st.session_state.dark

theme_color = {
    True: {
        "bg": "#1b1b1b",
        "text": "white",
        "card": "#2a2a2a",
        "border": "#c49b63",
    },
    False: {
        "bg": "white",
        "text": "black",
        "card": "#fffdf7",
        "border": "#d2b48c",
    }
}

css = f"""
<style>
body {{
    background-color: {theme_color[st.session_state.dark]['bg']} !important;
    color: {theme_color[st.session_state.dark]['text']} !important;
}}
.block-container {{
    background-color: {theme_color[st.session_state.dark]['bg']} !important;
    color: {theme_color[st.session_state.dark]['text']} !important;
}}
.card {{
    background: {theme_color[st.session_state.dark]['card']};
    padding: 20px;
    border-radius: 16px;
    border: 2px solid {theme_color[st.session_state.dark]['border']};
}}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Toggle button
st.sidebar.button("🌓 Toggle Dark Mode", on_click=switch_theme)

# ------------------------------------------------------
# GEMINI CONFIG
# ------------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini = genai.GenerativeModel("gemini-2.0-flash")

# ------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_breed_resnet.keras")

model = load_model()

# ------------------------------------------------------
# LOAD JSON FILES
# ------------------------------------------------------
@st.cache_data
def load_labels():
    return {int(v): k for k, v in json.load(open("class_indices.json")).items()}

@st.cache_data
def load_breed_info():
    data = json.load(open("120_breeds_new.json"))
    return {item["Breed"]: item for item in data}

label_map = load_labels()
breed_info = load_breed_info()

# ------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------
def normalize(s):
    return re.sub(r'[^a-z0-9]', "", s.lower())

# ------------------------------------------------------
# BREED DETAILS
# ------------------------------------------------------
def get_breed_details(pred_breed):
    clean = normalize(pred_breed)

    for breed in breed_info:
        if normalize(breed) == clean:
            return breed_info[breed]

    for breed in breed_info:
        if clean in normalize(breed):
            return breed_info[breed]

    return None

# ------------------------------------------------------
# PREDICT BREED (unchanged)
# ------------------------------------------------------
def predict_breed(img):
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)

    pred = model.predict(arr)
    idx = int(np.argmax(pred))
    return label_map[idx], float(np.max(pred) * 100)

# ------------------------------------------------------
# CHATBOT — Streamlit-Native Popup
# ------------------------------------------------------
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

def toggle_chat():
    st.session_state.chat_open = not st.session_state.chat_open

st.sidebar.button("🐶 Open Chatbot", on_click=toggle_chat)

if st.session_state.chat_open:
    st.sidebar.markdown("### 🐾 Dog AI Chatbot")
    msg = st.sidebar.text_input("Ask me anything about dogs:")
    if st.sidebar.button("Send"):
        if msg.strip():
            reply = gemini.generate_content(msg).text
            st.sidebar.write("**AI:** ", reply)

# ------------------------------------------------------
# HISTORY
# ------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------
page = st.sidebar.radio("Navigate", ["🏠 Home", "🐶 Breed Detector", "📜 History"])

# ------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------
if page == "🏠 Home":
    st.title("🐾 Dog Breed Detection System")
    st.markdown(
        """
        ### Welcome!  
        - 🧠 CNN model trained on 120 dog breeds  
        - 📸 Real-time breed prediction  
        - 💬 Gemini-powered chatbot  
        - 📘 Detailed breed info  
        - 📝 Prediction history  
        """
    )

# ------------------------------------------------------
# DETECTOR PAGE
# ------------------------------------------------------
elif page == "🐶 Breed Detector":
    st.title("🐶 Dog Breed Detector")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    upload = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

    if upload:
        img = Image.open(upload).convert("RGB")
        st.image(img, width=350)

        breed, conf = predict_breed(img)
        st.success(f"### 🐾 Breed: {breed}")
        st.info(f"### 📊 Confidence: {conf:.2f}%")

        st.session_state.history.append({
            "image": img,
            "breed": breed,
            "conf": conf
        })

        # KNOW MORE — FIXED
        if st.button("Know More 🐾"):
            info = get_breed_details(breed)
            if info:
                st.markdown(f"## 📘 About {breed}")
                for k, v in info.items():
                    if k != "Breed":
                        st.write(f"**{k}:** {v}")
            else:
                st.warning("No info available.")

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------
# HISTORY PAGE
# ------------------------------------------------------
elif page == "📜 History":
    st.title("📜 Prediction History")

    if not st.session_state.history:
        st.info("No predictions yet.")
    else:
        for h in st.session_state.history:
            st.image(h["image"], width=180)
            st.write(f"**Breed:** {h['breed']}")
            st.write(f"**Confidence:** {h['conf']:.2f}%")
            st.markdown("---")
