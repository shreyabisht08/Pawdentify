import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
from PIL import Image
import json
import base64

# -------------------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------------------
st.set_page_config(
    page_title="Dog Breed Detection",
    layout="wide"
)

# -------------------------------------------------------------
# CUSTOM BACKGROUND IMAGE FOR HOME PAGE
# -------------------------------------------------------------
def set_background(image_url):
    st.markdown(f"""
        <style>
        .main {{
            background-image: url('{image_url}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# DOG THEME + UI CSS
# -------------------------------------------------------------
dog_css = """
<style>
h1, h2, h3 {
    color: #3c2f2f;
    font-weight: 700;
}

/* Sidebar */
.css-1d391kg, .css-1lcbmhc {
    background-color: #fff7ec !important;
}

/* Cards */
.dog-card {
    padding: 20px;
    border-radius: 15px;
    background: #fffaf0;
    border: 2px solid #d2b48c;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.1);
}

/* Upload box */
.uploadbox {
    padding: 20px;
    background: #fff8e7;
    border: 2px dashed #c9a46c;
    border-radius: 15px;
    text-align: center;
}

/* FLOATING CHAT ICON */
#chat-icon {
    position: fixed;
    bottom: 35px;
    right: 35px;
    width: 80px;
    height: 80px;
    background-image: url('https://i.imgur.com/8qZz7oZ.png');
    background-size: cover;
    background-position: center;
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.3);
    animation: float 3s infinite ease-in-out;
    z-index: 99999;
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-12px); }
    100% { transform: translateY(0px); }
}

/* CHAT POPUP WINDOW */
#chat-popup {
    position: fixed;
    bottom: 130px;
    right: 30px;
    width: 340px;
    height: 480px;
    background: white;
    border-radius: 20px;
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

#chat-close {
    float: right;
    cursor: pointer;
    color: #885533;
    font-size: 22px;
}
</style>
"""
st.markdown(dog_css, unsafe_allow_html=True)

# -------------------------------------------------------------
# CHATBOT POPUP JAVASCRIPT
# -------------------------------------------------------------
chat_js = """
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
st.markdown(chat_js, unsafe_allow_html=True)

# -------------------------------------------------------------
# GEMINI API CONFIG
# -------------------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_breed_resnet.keras")

model = load_model()

# -------------------------------------------------------------
# LOAD LABEL MAP
# -------------------------------------------------------------
@st.cache_data
def load_classes():
    with open("class_indices.json", "r") as f:
        data = json.load(f)
    return {int(v): k for k, v in data.items()}

label_map = load_classes()

# -------------------------------------------------------------
# BREED INFO
# -------------------------------------------------------------
@st.cache_data
def load_breed_info():
    with open("120_breeds_new.json", "r") as f:
        data = json.load(f)
    return {item["Breed"]: item for item in data}

breed_info = load_breed_info()

# -------------------------------------------------------------
# PREDICT BREED FUNCTION
# -------------------------------------------------------------
def predict_breed(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr)
    idx = np.argmax(preds)
    breed = label_map[idx]
    conf = float(np.max(preds) * 100)
    return breed, conf

# -------------------------------------------------------------
# APP SIDEBAR
# -------------------------------------------------------------
pages = ["🏠 Home", "🐶 Breed Detector", "📂 Prediction History"]
choice = st.sidebar.radio("Navigate", pages)

# -------------------------------------------------------------
# 1️⃣ HOME PAGE
# -------------------------------------------------------------
if choice == "🏠 Home":
    set_background("https://wallpapercave.com/wp/wp2704827.jpg")

    st.title("🐾 Dog Breed Detection System")

    st.markdown("""
    This application includes:

    - 🧠 Deep learning model trained on **120 dog breeds**
    - 📸 Real-time dog breed prediction
    - 🤖 AI chatbot powered by **Gemini**
    - 📘 Detailed breed information
    - 📝 History of all predictions
    """)

# -------------------------------------------------------------
# 2️⃣ BREED DETECTOR
# -------------------------------------------------------------
elif choice == "🐶 Breed Detector":
    st.title("🐶 Upload an Image to Detect Breed")
    st.markdown('<div class="dog-card">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Dog Image", type=["jpg", "jpeg", "png", "webp"])

    if "history" not in st.session_state:
        st.session_state.history = []

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=300)

        breed, conf = predict_breed(img)

        st.success(f"### 🐾 Breed: **{breed}**")
        st.info(f"### 📊 Confidence: **{conf:.2f}%**")

        # Save to history
        st.session_state.history.append({
            "image": uploaded.getvalue(),
            "breed": breed,
            "confidence": conf
        })

        if st.button("Know More 🐾"):
            st.markdown("### 📘 Breed Details")
            info = breed_info.get(breed, None)
            if info:
                for key, value in info.items():
                    if key != "Breed":
                        st.write(f"**{key}:** {value}")
            else:
                st.warning("No details available.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 3️⃣ PREDICTION HISTORY
# -------------------------------------------------------------
elif choice == "📂 Prediction History":
    st.title("📂 Your Past Predictions")

    if "history" not in st.session_state or len(st.session_state.history) == 0:
        st.info("No history yet.")
    else:
        for item in st.session_state.history:
            st.image(item["image"], width=200)
            st.write(f"**Breed:** {item['breed']}")
            st.write(f"**Confidence:** {item['confidence']:.2f}%")
            st.markdown("---")

# -------------------------------------------------------------
# CHATBOT POPUP HTML
# -------------------------------------------------------------
st.markdown("""
<div id="chat-popup">
    <span id="chat-close" onclick="toggleChat()">✖</span>
    <h4>🤖 PawBot</h4>
    <p style="font-size:13px;">Ask me anything about dogs!</p>
</div>

<div id="chat-icon" onclick="toggleChat()"></div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# CHATBOT LOGIC
# -------------------------------------------------------------
chat_msg = st.text_input("", key="chat_input", label_visibility="collapsed")

if st.button("Send", key="send_chat"):
    if chat_msg.strip():
        reply = gemini.generate_content(chat_msg)
        st.write("**PawBot:**", reply.text)
