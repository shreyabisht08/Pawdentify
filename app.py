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
# CUSTOM CSS (Dog Theme + Popup Chatbot + Background)
# ------------------------------------------------------
custom_css = """
<style>

body, .main {
    font-family: 'Poppins', sans-serif;
}

/* HOME PAGE BACKGROUND */
.home-background {
    background-image: url('https://images.unsplash.com/photo-1517849845537-4d257902454a');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    padding: 80px;
    border-radius: 18px;
    position: relative;
}

.home-background::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(255,255,255,0.75);
    border-radius: 18px;
    backdrop-filter: blur(3px);
}

.home-content {
    position: relative;
    z-index: 10;
}

/* DOG CARD */
.dog-card {
    padding: 20px;
    border-radius: 15px;
    background: #fffdf7;
    border: 2px solid #d2b48c;
    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
}

/* CHATBOT POPUP */
#chat-popup {
    position: fixed;
    bottom: 110px;
    right: 35px;
    width: 340px;
    height: 460px;
    background: #ffffff;
    border-radius: 15px;
    border: 2px solid #bb9b72;
    padding: 0;
    display: none;
    box-shadow: 0px 6px 28px rgba(0,0,0,0.25);
    animation: slideUp 0.4s ease-out forwards;
    z-index: 999999;
}

@keyframes slideUp {
    from { transform: translateY(80px); opacity: 0; }
    to   { transform: translateY(0); opacity: 1; }
}

/* CHATBOT HEADER */
.chat-header {
    background: #7b573c;
    color: white;
    padding: 15px;
    border-radius: 12px 12px 0 0;
    font-size: 18px;
    font-weight: 600;
}

#chat-close {
    float: right;
    cursor: pointer;
    color: white;
    font-size: 20px;
}

/* CHATBOT CONTENT */
.chat-body {
    padding: 15px;
    height: 330px;
    overflow-y: auto;
}

/* CHATBOT INPUT BOX */
.chat-input-box {
    padding: 10px;
    border-top: 1px solid #d8c2a6;
}

/* FLOATING CHAT ICON */
#chat-btn {
    position: fixed;
    bottom: 35px;
    right: 35px;
    background: #8b5e34;
    color: white;
    border-radius: 50px;
    padding: 14px;
    font-size: 22px;
    cursor: pointer;
    border: none;
    z-index: 99999;
    transition: 0.3s;
}
#chat-btn:hover { background: #5c3b21; }

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# JS for chatbot toggle
toggle_js = """
<script>
function toggleChat() {
    const chat = document.getElementById("chat-popup");
    chat.style.display = chat.style.display === "block" ? "none" : "block";
}
</script>
"""
st.markdown(toggle_js, unsafe_allow_html=True)

# ------------------------------------------------------
# GEMINI CONFIG
# ------------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini = genai.GenerativeModel("gemini-2.0-flash")

# ------------------------------------------------------
# LOAD MODEL + JSON DATA
# ------------------------------------------------------
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
def load_info():
    with open("120_breeds_new.json", "r") as f:
        data = json.load(f)
    return {item["Breed"]: item for item in data}

breed_info = load_info()

# ------------------------------------------------------
# CLEAN + SMART MATCHING FOR KNOW MORE
# ------------------------------------------------------
def clean_name(name):
    return re.sub(r'[^a-z0-9 ]', '', name.lower().replace("_", " ").strip())

def get_breed_details(predicted_breed):
    pred_clean = clean_name(predicted_breed)

    # Exact clean match
    for breed in breed_info:
        if clean_name(breed) == pred_clean:
            return format_details(breed_info[breed], breed)

    # Partial match (German Shepherd vs German Shepherd Dog)
    for breed in breed_info:
        if pred_clean in clean_name(breed):
            return format_details(breed_info[breed], breed)

    return "No details available."

def format_details(info, breed):
    text = f"### 📘 About {breed}\n"
    for k, v in info.items():
        if k != "Breed":
            text += f"**{k}:** {v}\n\n"
    return text

# ------------------------------------------------------
# PREDICT BREED (UNCHANGED)
# ------------------------------------------------------
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

# ------------------------------------------------------
# HISTORY
# ------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------
page = st.sidebar.radio("Navigate", ["🏠 Home", "🐶 Breed Detector", "📜 Prediction History"])

# ------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------
if page == "🏠 Home":
    st.markdown('<div class="home-background"><div class="home-content">', unsafe_allow_html=True)

    st.title("🐾 Dog Breed Detection System")
    st.write("""
    This application includes:

    - 🧠 AI model trained on 120 dog breeds  
    - 📸 Real-time dog breed prediction  
    - 🤖 Dog chatbot powered by Gemini  
    - 📘 Detailed breed knowledge  
    - 📝 Prediction history  
    """)

    st.markdown('</div></div>', unsafe_allow_html=True)

# ------------------------------------------------------
# BREED DETECTOR PAGE
# ------------------------------------------------------
elif page == "🐶 Breed Detector":
    st.title("🐶 Dog Breed Detector")
    st.markdown('<div class="dog-card">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a dog image", type=["jpg","jpeg","png","webp"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=300)

        breed, conf = predict_breed(img)
        st.success(f"### 🐾 Breed: **{breed}**")
        st.info(f"### 📊 Confidence: **{conf:.2f}%**")

        st.session_state.history.append({"image": img, "breed": breed, "conf": conf})

        if st.button("Know More 🐾"):
            st.markdown(get_breed_details(breed))

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------
# HISTORY PAGE
# ------------------------------------------------------
elif page == "📜 Prediction History":
    st.title("📜 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet.")
    else:
        for h in st.session_state.history:
            st.image(h["image"], width=200)
            st.write(f"**Breed:** {h['breed']}")
            st.write(f"**Confidence:** {h['conf']:.2f}%")
            st.markdown("---")

# ------------------------------------------------------
# CHATBOT POPUP - Hidden until clicked
# ------------------------------------------------------
st.markdown("""
<div id="chat-popup">
    <div class="chat-header">
        🐶 Dog Chatbot
        <span id="chat-close" onclick="toggleChat()">✖</span>
    </div>

    <div class="chat-body" id="chat-body">
        <p><b>👋 Hi! I'm your Dog Assistant!</b></p>
        <p>Ask me anything about dogs.</p>
    </div>

    <div class="chat-input-box">
""", unsafe_allow_html=True)

popup_input = st.text_input("Message", key="chat_input_popup")

if st.button("Send", key="chat_send_popup"):
    if popup_input.strip():
        reply = gemini.generate_content(popup_input).text
        st.markdown(f"**You:** {popup_input}")
        st.markdown(f"**AI:** {reply}")

st.markdown("""
    </div>
</div>

<button id="chat-btn" onclick="toggleChat()">💬</button>
""", unsafe_allow_html=True)
