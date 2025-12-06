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
# CUSTOM CSS (Dog Theme + Background + Popup Chatbot)
# ------------------------------------------------------
custom_css = """
<style>

body, .main {
    font-family: 'Poppins', sans-serif;
}

/* ---------- HOME PAGE BACKGROUND ---------- */
.home-background {
    background-image: url('https://images.unsplash.com/photo-1517849845537-4d257902454a');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    padding: 100px;
    border-radius: 20px;
    position: relative;
}

.home-background::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(255,255,255,0.75);
    border-radius: 20px;
    backdrop-filter: blur(2px);
}

.home-content {
    position: relative;
    z-index: 5;
}

/* ---------- Doggy Card ---------- */
.dog-card {
    padding: 20px;
    border-radius: 15px;
    background: #fffdf7;
    border: 2px solid #d2b48c;
    box-shadow: 0 4px 18px rgba(0,0,0,0.1);
}

/* ---------- Popup Chatbot Box ---------- */
#chat-popup {
    position: fixed;
    bottom: 120px;
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

/* ---------- Floating Chat Button ---------- */
#chat-btn {
    position: fixed;
    bottom: 35px;
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

/* Close button */
#chat-close {
    float: right;
    cursor: pointer;
    color: #8b5e34;
    font-size: 20px;
}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Chat toggle JS
toggle_js = """
<script>
function toggleChat() {
    const box = document.getElementById("chat-popup");
    box.style.display = box.style.display === "block" ? "none" : "block";
}
</script>
"""
st.markdown(toggle_js, unsafe_allow_html=True)

# ------------------------------------------------------
# GEMINI AI CONFIG
# ------------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini = genai.GenerativeModel("gemini-2.0-flash")

# ------------------------------------------------------
# LOAD MODEL + DATA
# ------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dog_breed_resnet.keras")

model = load_model()

@st.cache_data
def load_labels():
    with open("class_indices.json") as f:
        classes = json.load(f)
    return {int(v): k for k, v in classes.items()}

label_map = load_labels()

@st.cache_data
def load_info():
    with open("120_breeds_new.json", "r") as f:
        data = json.load(f)
    return {item["Breed"]: item for item in data}

breed_info = load_info()

# ------------------------------------------------------
# PREDICT BREED
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
# BREED DETAILS
# ------------------------------------------------------
def get_breed_details(breed):
    if breed in breed_info:
        info = breed_info[breed]
        text = f"### 📘 About {breed}\n"
        for k, v in info.items():
            if k != "Breed":
                text += f"**{k}:** {v}\n\n"
        return text
    return "No details available."

# ------------------------------------------------------
# HISTORY MEMORY
# ------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ------------------------------------------------------
# SIDEBAR NAV
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

    - 🧠 A deep learning model trained on 120 dog breeds  
    - 📸 Real-time dog breed prediction  
    - 🤖 AI chatbot powered by Gemini  
    - 📘 Detailed breed information  
    - 📝 History of predictions  
    """)

    st.markdown('</div></div>', unsafe_allow_html=True)

# ------------------------------------------------------
# BREED DETECTOR PAGE
# ------------------------------------------------------
elif page == "🐶 Breed Detector":
    st.title("🐶 Dog Breed Detector")

    st.markdown('<div class="dog-card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png", "webp"])

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
# POPUP CHATBOT (Hidden by default)
# ------------------------------------------------------
st.markdown("""
<div id="chat-popup">
    <span id="chat-close" onclick="toggleChat()">✖</span>
    <h4>🐾 Dog AI Chatbot</h4>
    <p style="font-size:13px;">Ask anything about dogs!</p>
</div>

<button id="chat-btn" onclick="toggleChat()">💬 Chat</button>
""", unsafe_allow_html=True)

# Chat input only inside popup
popup_query = st.text_input("", key="chat_input_popup", label_visibility="collapsed")
if st.button("Send", key="chat_send_popup"):
    if popup_query.strip():
        reply = gemini.generate_content(popup_query)
        st.write("**AI:**", reply.text)
