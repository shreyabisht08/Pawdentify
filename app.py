import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import google.generativeai as genai
import json

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Dog Breed Detector", layout="wide")

# ------------------------------------------------------
# DOG-THEME + ANIMATED CHATBOT CSS
# ------------------------------------------------------
dog_theme = """
<style>

body {
    font-family: 'Poppins', sans-serif;
}

/* Background */
.main {
    background: linear-gradient(135deg, #fefcfb 0%, #f6e6d4 100%);
}

/* Headers */
h1, h2, h3 {
    color: #3c2f2f;
    font-weight: 700;
}

/* Dog card */
.dog-card {
    padding: 20px;
    border-radius: 15px;
    background: #fffaf0;
    border: 2px solid #d2b48c;
    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
}

/* Upload box */
.uploadbox {
    padding: 20px;
    background: #fff8e7;
    border: 2px dashed #c9a46c;
    border-radius: 15px;
    text-align: center;
    transition: 0.3s;
}
.uploadbox:hover {
    border-color: #8b5e34;
    background: #fff2cd;
}

/* --- Animated popup chatbot --- */
#chat-popup {
    position: fixed;
    bottom: 110px;
    right: 35px;
    width: 330px;
    height: 450px;
    background: white;
    border-radius: 15px;
    border: 2px solid #c9a46c;
    padding: 15px;
    display: none;
    box-shadow: 0px 6px 28px rgba(0,0,0,0.25);
    animation: slideUp 0.4s ease-out forwards;
    z-index: 99999;
}

/* Animation */
@keyframes slideUp {
  from { transform: translateY(60px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

/* Chat button */
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
#chat-btn:hover {
    background: #5c3b21;
}

/* Close button */
#chat-close {
    float: right;
    cursor: pointer;
    color: #8b5e34;
    font-size: 20px;
}

</style>
"""
st.markdown(dog_theme, unsafe_allow_html=True)

# ------------------------------------------------------
# POPUP CHATBOT JAVASCRIPT
# ------------------------------------------------------
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
# GEMINI CONFIG
# ------------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini = genai.GenerativeModel("gemini-2.5-flash")

# ------------------------------------------------------
# LOAD DOG BREED MODEL
# ------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_breed_resnet.keras")

model = load_model()

# ------------------------------------------------------
# LOAD CLASS INDICES
# ------------------------------------------------------
@st.cache_data
def load_class_indices():
    with open("class_indices.json") as f:
        data = json.load(f)
    return {int(v): k for k, v in data.items()}

label_map = load_class_indices()

# ------------------------------------------------------
# LOAD BREED INFO FROM JSON
# ------------------------------------------------------
@st.cache_data
def load_breed_info():
    with open("120_breeds_new.json", "r") as f:
        data = json.load(f)
    return {item["Breed"]: item for item in data}

breed_info = load_breed_info()

# ------------------------------------------------------
# PREDICT BREED
# ------------------------------------------------------
def predict_breed(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = np.argmax(preds)
    breed = label_map[idx]
    conf = float(np.max(preds) * 100)
    return breed, conf

# ------------------------------------------------------
# GET BREED DETAILS
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
# SIDEBAR NAVIGATION
# ------------------------------------------------------
pages = ["🏠 Home", "🐶 Dog Breed Detector"]
choice = st.sidebar.radio("Navigation", pages)

# ------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------
if choice == "🏠 Home":
    st.title("🐾 Dog Breed Detection System")
    st.write("""
    ### Welcome!  
    This project combines:
    - 🧠 A CNN model trained on 120 dog breeds  
    - 🔍 A real-time breed detection system  
    - 🤖 Gemini AI chatbot for dog-related questions  
    - 📘 Detailed breed information from curated JSON  
    """)

    st.image("https://i.imgur.com/CuQZ8kC.jpeg", width=400)

    st.markdown("Use the sidebar to begin!")

# ------------------------------------------------------
# DOG BREED DETECTOR PAGE
# ------------------------------------------------------
elif choice == "🐶 Dog Breed Detector":
    st.title("🐶 Dog Breed Detector")
    st.markdown('<div class="dog-card">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a Dog Image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=300)

        breed, conf = predict_breed(img)
        st.success(f"### 🐾 Predicted Breed: **{breed}**")
        st.info(f"### 📊 Confidence: **{conf:.2f}%**")

        if st.button("Know More 🐾"):
            st.markdown(get_breed_details(breed))

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------
# POPUP CHATBOT UI (HTML)
# ------------------------------------------------------
st.markdown("""
<div id="chat-popup">
    <span id="chat-close" onclick="toggleChat()">✖</span>
    <h4>🐾 Ask Shreya's Dog AI</h4>
    <p style="font-size:13px;">Ask anything about dogs — training, care, breeds…</p>
</div>

<button id="chat-btn" onclick="toggleChat()">💬 Chat</button>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# CHATBOT INPUT + RESPONSE
# ------------------------------------------------------
chat_input = st.text_input("Chat here:", key="popup_input")
if st.button("Send", key="popup_send"):
    if chat_input.strip():
        reply = gemini.generate_content(chat_input)
        st.write("**AI:**", reply.text)
