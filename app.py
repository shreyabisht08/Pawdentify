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
# DARK MODE TOGGLE
# ------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

st.sidebar.markdown("### 🌓 Theme")
st.sidebar.button("Toggle Dark / Light Mode", on_click=toggle_dark_mode)

# ------------------------------------------------------
# COLOR SYSTEM CSS (LIGHT + DARK)
# ------------------------------------------------------
light_css = """
<style>
:root {
    --bg-main: #ffffff;
    --card-bg: #fffdf7;
    --border-color: #d2b48c;
    --heading-color: #3c2f2f;
    --text-color: #333;
    --chat-header: #8b5e34;
    --chat-bg: #fffaf2;
    --user-bubble: #d1e7ff;
    --ai-bubble: #ffe8c6;
}
</style>
"""

dark_css = """
<style>
:root {
    --bg-main: #1c1c1c;
    --card-bg: #2a2a2a;
    --border-color: #c49b63;
    --heading-color: #f5deb3;
    --text-color: #e6e6e6;
    --chat-header: #b8860b;
    --chat-bg: #252525;
    --user-bubble: #4a76a8;
    --ai-bubble: #8b6f47;
}
</style>
"""

st.markdown(dark_css if st.session_state.dark_mode else light_css, unsafe_allow_html=True)

# ------------------------------------------------------
# MAIN UI CSS
# ------------------------------------------------------
custom_css = """
<style>

body, .main {
    font-family: 'Poppins', sans-serif;
    background: var(--bg-main);
    color: var(--text-color);
}

/*************** HOME BACKGROUND ***************/
.home-bg {
    background-image: url('https://images.unsplash.com/photo-1517849845537-4d257902454a');
    background-size: cover;
    padding: 100px;
    border-radius: 18px;
    position: relative;
}
.home-bg::before {
    content: "";
    position: absolute;
    inset: 0;
    background: rgba(255,255,255,0.78);
    border-radius: 18px;
    backdrop-filter: blur(3px);
}
.home-content { position: relative; z-index: 5; }

/*************** DOG CARD ***************/
.dog-card {
    background: var(--card-bg);
    padding: 25px;
    border-radius: 16px;
    border: 2px solid var(--border-color);
    box-shadow: 0 4px 18px rgba(0,0,0,0.15);
}

/*************** CHATBOT POPUP ***************/
#chatbot-box {
    position: fixed;
    bottom: 120px;
    right: 35px;
    width: 360px;
    height: 500px;
    background: var(--card-bg);
    border: 2px solid var(--border-color);
    border-radius: 18px;
    display: none;
    box-shadow: 0 8px 28px rgba(0,0,0,0.3);
    animation: slideIn .4s ease-out forwards;
    z-index: 999999;
}

@keyframes slideIn {
    from { transform: translateY(80px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

.chat-header {
    background: var(--chat-header);
    padding: 15px;
    color: white;
    font-size: 18px;
    border-radius: 16px 16px 0 0;
}

.chat-body {
    height: 360px;
    padding: 15px;
    overflow-y: auto;
    background: var(--chat-bg);
}

/*************** CHAT BUBBLES ***************/
.user-bubble {
    background: var(--user-bubble);
    padding: 10px 14px;
    border-radius: 20px;
    margin: 8px 0;
    width: fit-content;
}

.ai-bubble {
    background: var(--ai-bubble);
    padding: 10px 14px;
    border-radius: 20px;
    margin: 8px 0;
    width: fit-content;
}

/*************** INPUT AREA ***************/
.chat-input-area {
    padding: 10px;
    background: var(--bg-main);
    border-top: 2px solid var(--border-color);
}
.chat-input {
    width: 75%;
    padding: 8px;
    background: var(--card-bg);
    border-radius: 10px;
    color: var(--text-color);
    border: 1px solid var(--border-color);
}

/*************** CHAT ICON ***************/
#chatbot-icon {
    position: fixed;
    bottom: 35px;
    right: 35px;
    width: 70px;
    cursor: pointer;
    z-index: 999999;
}

</style>

<script>
function toggleChat() {
    var x = document.getElementById("chatbot-box");
    x.style.display = (x.style.display === "block") ? "none" : "block";
}
</script>
"""

st.markdown(custom_css, unsafe_allow_html=True)

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
# LOAD JSON DATA
# ------------------------------------------------------
@st.cache_data
def load_labels():
    with open("class_indices.json") as f:
        data = json.load(f)
    return {int(v): k for k, v in data.items()}

label_map = load_labels()

@st.cache_data
def load_info():
    with open("120_breeds_new.json") as f:
        data = json.load(f)
    return {item["Breed"]: item for item in data}

breed_info = load_info()

# ------------------------------------------------------
# SMART BREED MATCHING
# ------------------------------------------------------
def normalize(n):
    return re.sub(r'[^a-z0-9]', "", n.lower())

def format_info(breed):
    info = breed_info[breed]
    text = f"### 📘 About {breed}\n\n"
    for k, v in info.items():
        if k != "Breed":
            text += f"**{k}:** {v}\n\n"
    return text

def get_breed_details(pred_breed):
    clean = normalize(pred_breed)

    # exact match
    for breed in breed_info:
        if normalize(breed) == clean:
            return format_info(breed)

    # partial match
    for breed in breed_info:
        if clean in normalize(breed):
            return format_info(breed)

    return "No details available."

# ------------------------------------------------------
# PREDICT BREED (your logic unchanged)
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
# HISTORY
# ------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------
page = st.sidebar.radio("Navigate", ["🏠 Home", "🐶 Breed Detector", "📜 History"])

# ------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------
if page == "🏠 Home":
    st.markdown('<div class="home-bg"><div class="home-content">', unsafe_allow_html=True)
    st.title("🐾 Dog Breed Detection System")
    st.write("""
    - 🧠 Trained on 120 dog breeds  
    - 📸 Real-time dog breed detection  
    - 🐕 Detailed breed information  
    - 💬 Dog Chatbot powered by Gemini  
    - 📝 Prediction History  
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
        st.info(f"### 📊 Confidence: {conf:.2f}%")

        st.session_state.history.append({"image": img, "breed": breed, "conf": conf})

        if st.button("Know More 🐾"):
            st.markdown(get_breed_details(breed))

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
            st.image(h["image"], width=200)
            st.write(f"**Breed:** {h['breed']}")
            st.write(f"**Confidence:** {h['conf']:.2f}%")
            st.markdown("---")

# ------------------------------------------------------
# CHATBOT POPUP
# ------------------------------------------------------
st.markdown("""
<img id="chatbot-icon" src="https://cdn-icons-png.flaticon.com/512/4712/4712100.png" onclick="toggleChat()">

<div id="chatbot-box">
    <div class="chat-header">
        🐶 Dog Chatbot
        <span onclick="toggleChat()" style="float:right; cursor:pointer;">✖</span>
    </div>

    <div class="chat-body" id="chat-body"></div>

    <div class="chat-input-area">
        <input id="chat-msg" class="chat-input" placeholder="Ask something...">
        <button onclick="sendMsg()">Send</button>
    </div>
</div>

<script>
function sendMsg() {
    let msg = document.getElementById("chat-msg").value;
    if (msg.trim() === "") return;

    document.getElementById("chat-body").innerHTML +=
      '<div class="user-bubble"><b>You:</b> ' + msg + '</div>';

    const hidden = document.createElement("textarea");
    hidden.name = "hidden_msg";
    hidden.value = msg;
    hidden.style.display = "none";
    document.body.appendChild(hidden);

    document.getElementById("hidden-btn").click();
    document.getElementById("chat-msg").value = "";
}
</script>
""", unsafe_allow_html=True)

# Hidden Streamlit callback
hidden = st.button("SEND", key="hidden-btn")

if hidden:
    msg = st.session_state.get("hidden_msg", "")
    if msg:
        reply = gemini.generate_content(msg).text
        st.session_state["bot_reply"] = reply

if "bot_reply" in st.session_state:
    r = st.session_state["bot_reply"]
    st.markdown(
        f"""
        <script>
        document.getElementById("chat-body").innerHTML +=
          '<div class="ai-bubble"><b>AI:</b> {r}</div>';
        </script>
        """,
        unsafe_allow_html=True
    )
