import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import google.generativeai as genai
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Dog Breed Detector", layout="wide")

# ------------------------------------------------------
# CUSTOM CSS (Dog Theme + Popup Chatbot)
# ------------------------------------------------------
dog_theme = """
<style>
body { font-family: 'Poppins', sans-serif; }
.main { background: linear-gradient(135deg, #fffaf4 0%, #ffe8d6 100%); }

h1, h2, h3 { color: #4b3621; font-weight: 700; }

.dog-card {
    padding: 20px;
    border-radius: 15px;
    background: #fffdf7;
    border: 2px solid #d2b48c;
    box-shadow: 0 4px 18px rgba(0,0,0,0.1);
}

#chat-popup {
    position: fixed;
    bottom: 110px;
    right: 35px;
    width: 330px;
    height: 430px;
    background: white;
    border-radius: 15px;
    border: 2px solid #c9a46c;
    padding: 15px;
    display: none;
    box-shadow: 0px 6px 28px rgba(0,0,0,0.25);
    animation: slideUp 0.4s ease-out forwards;
    z-index: 99999;
}

@keyframes slideUp {
    from { transform: translateY(60px); opacity: 0; }
    to   { transform: translateY(0); opacity: 1; }
}

#chat-btn {
    position: fixed;
    bottom: 30px;
    right: 35px;
    background: #8b5e34;
    color: white;
    border-radius: 40px;
    padding: 14px 24px;
    font-size: 17px;
    font-weight: bold;
    cursor: pointer;
    border: none;
    z-index: 99999;
    transition: 0.3s;
}
#chat-btn:hover { background: #5c3b21; }

#chat-close {
    float: right;
    cursor: pointer;
    color: #8b5e34;
    font-size: 20px;
}
</style>
"""
st.markdown(dog_theme, unsafe_allow_html=True)

# JS for popup chatbot
popup_js = """
<script>
function toggleChat() {
    var box = document.getElementById("chat-popup");
    if (box.style.display === "none") {
        box.style.display = "block";
    } else {
        box.style.display = "none";
    }
}
</script>
"""
st.markdown(popup_js, unsafe_allow_html=True)

# ------------------------------------------------------
# GEMINI AI CONFIG
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
# LOAD CLASS LABELS
# ------------------------------------------------------
@st.cache_data
def load_class_indices():
    with open("class_indices.json", "r") as f:
        data = json.load(f)
    return {int(v): k for k, v in data.items()}

label_map = load_class_indices()

# ------------------------------------------------------
# LOAD BREED DETAILS
# ------------------------------------------------------
@st.cache_data
def load_breed_info():
    with open("120_breeds_new.json", "r") as f:
        data = json.load(f)
    return {item["Breed"]: item for item in data}

breed_info = load_breed_info()

# ------------------------------------------------------
# BREED PREDICTION (FIXED VERSION)
# ------------------------------------------------------
def predict_breed(img):
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)

    # ⭐ CRITICAL FIX ⭐
    arr = preprocess_input(arr)

    preds = model.predict(arr)
    idx = np.argmax(preds)
    breed = label_map[idx]
    conf = float(np.max(preds) * 100)
    return breed, conf

# ------------------------------------------------------
# BREED DETAILS
# ------------------------------------------------------
def get_breed_details(breed):
    if breed in breed_info:
        info = breed_info[breed]
        output = f"### 📘 About {breed}\n"
        for k, v in info.items():
            if k != "Breed":
                output += f"**{k}:** {v}\n\n"
        return output
    return "No information available."

# ------------------------------------------------------
# SESSION STATE (for Prediction History)
# ------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------
pages = ["🏠 Home", "🐶 Breed Detector", "📜 Prediction History"]
choice = st.sidebar.radio("Navigate", pages)

# ------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------
if choice == "🏠 Home":
    st.title("🐾 Dog Breed Detection System")
    st.write("""
    This application includes:
    - 🧠 A deep learning model trained on 120 dog breeds  
    - 📷 Real-time dog breed prediction  
    - 🤖 AI chatbot powered by Gemini  
    - 📘 Detailed breed information  
    - 📝 History of all predictions  
    """)

# ------------------------------------------------------
# BREED DETECTOR PAGE
# ------------------------------------------------------
elif choice == "🐶 Breed Detector":
    st.title("🐶 Dog Breed Detector")
    st.markdown('<div class="dog-card">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=300)

        breed, conf = predict_breed(img)

        st.success(f"### 🐾 Breed: **{breed}**")
        st.info(f"### 📊 Confidence: **{conf:.2f}%**")

        # save to history
        st.session_state["history"].append({
            "image": img,
            "breed": breed,
            "confidence": conf
        })

        if st.button("Know More 🐾"):
            st.markdown(get_breed_details(breed))

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------
# HISTORY PAGE
# ------------------------------------------------------
elif choice == "📜 Prediction History":
    st.title("📜 Past Predictions")

    if len(st.session_state["history"]) == 0:
        st.info("No predictions yet!")
    else:
        for idx, item in enumerate(st.session_state["history"]):
            st.write(f"### 🐶 Prediction #{idx+1}")
            st.image(item["image"], width=200)
            st.write(f"**Breed:** {item['breed']}")
            st.write(f"**Confidence:** {item['confidence']:.2f}%")
            st.markdown("---")

# ------------------------------------------------------
# POPUP CHATBOT WINDOW
# ------------------------------------------------------
st.markdown("""
<div id="chat-popup">
    <span id="chat-close" onclick="toggleChat()">✖</span>
    <h4>🐾 Dog Chatbot</h4>
    <p style="font-size:13px;">Ask anything about dogs!</p>
</div>
<button id="chat-btn" onclick="toggleChat()">💬 Chat</button>
""", unsafe_allow_html=True)

# CHATBOT INPUT
chat_input = st.text_input("Chatbot:", key="popup_input")
if st.button("Send", key="popup_send"):
    if chat_input.strip():
        reply = gemini.generate_content(chat_input)
        st.write("**AI:**", reply.text)
