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
st.set_page_config(page_title="Pawdentify – Dog Breed AI", layout="wide")

# ------------------------------------------------------
# DARK / LIGHT MODE TOGGLE
# ------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

st.sidebar.markdown("## ⚙️ Settings")
dark_mode = st.sidebar.toggle("🌙 Dark mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = dark_mode

# Colors based on mode
if dark_mode:
    BG_GRADIENT = "linear-gradient(135deg, #0f172a 0%, #111827 40%, #1f2933 100%)"
    CARD_BG = "#111827"
    CARD_BORDER = "#374151"
    TEXT_COLOR = "#f9fafb"
    SUBTEXT_COLOR = "#d1d5db"
    ACCENT = "#f97316"
    CHAT_USER_BG = "#22c55e"
    CHAT_BOT_BG = "#1f2937"
else:
    BG_GRADIENT = "linear-gradient(135deg, #fff7ed 0%, #fef3c7 40%, #e0f2fe 100%)"
    CARD_BG = "#ffffff"
    CARD_BORDER = "#e5e7eb"
    TEXT_COLOR = "#111827"
    SUBTEXT_COLOR = "#4b5563"
    ACCENT = "#f97316"
    CHAT_USER_BG = "#3b82f6"
    CHAT_BOT_BG = "#f3f4f6"

# ------------------------------------------------------
# GLOBAL CSS (THEME + CHAT STYLES)
# ------------------------------------------------------
custom_css = f"""
<style>
/* Global background */
.reportview-container, .main, body {{
    background: {BG_GRADIENT} !important;
    color: {TEXT_COLOR} !important;
    font-family: 'Poppins', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

/* Center main area and soften */
.block-container {{
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1150px;
}}

/* Sidebar */
section[data-testid="stSidebar"] > div {{
    background: rgba(15, 23, 42, 0.9) if {str(dark_mode).lower()} else "#ffffff";
    color: #e5e7eb;
}}

/* Title paw emoji */
.paw-title {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 2.3rem;
    font-weight: 700;
    color: {TEXT_COLOR};
}}
.paw-title span.paw-icon {{
    font-size: 2.6rem;
}}

/* Generic card */
.card {{
    background: {CARD_BG};
    border-radius: 18px;
    border: 1px solid {CARD_BORDER};
    padding: 1.5rem 1.7rem;
    box-shadow: 0 18px 35px rgba(15, 23, 42, 0.18);
}}

/* Upload box */
.upload-box {{
    border: 2px dashed {ACCENT};
    border-radius: 18px;
    padding: 1.5rem;
    background: rgba(255,255,255,0.04);
}}

/* Chat container */
.chat-wrapper {{
    height: 520px;
    border-radius: 18px;
    border: 1px solid {CARD_BORDER};
    background: {CARD_BG};
    padding: 1rem 1.2rem;
    display: flex;
    flex-direction: column;
}}

.chat-window {{
    flex: 1;
    overflow-y: auto;
    padding-right: 0.5rem;
}}

.chat-msg {{
    margin: 0.3rem 0;
    max-width: 80%;
    padding: 0.6rem 0.9rem;
    border-radius: 14px;
    font-size: 0.95rem;
    line-height: 1.4;
}}

.chat-user {{
    margin-left: auto;
    background: {CHAT_USER_BG};
    color: white;
    border-bottom-right-radius: 4px;
}}

.chat-bot {{
    margin-right: auto;
    background: {CHAT_BOT_BG};
    color: {TEXT_COLOR};
    border-bottom-left-radius: 4px;
}}

.chat-input-bar {{
    margin-top: 0.6rem;
    display: flex;
    gap: 0.5rem;
    align-items: center;
}}

.chat-input-bar input {{
    flex: 1;
}}

.small-tag {{
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    background: rgba(249,115,22,0.08);
    color: {ACCENT};
    font-size: 0.7rem;
    border: 1px solid rgba(249,115,22,0.4);
}}

.history-card {{
    background: {CARD_BG};
    border-radius: 16px;
    border: 1px solid {CARD_BORDER};
    padding: 0.7rem;
    box-shadow: 0 10px 20px rgba(15,23,42,0.18);
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

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
    with open("120_breeds_new.json") as f:
        data = json.load(f)
    return {item["Breed"]: item for item in data}

breed_info = load_info()

# ------------------------------------------------------
# BREED PREDICTION  (unchanged logic)
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
    return "No information available."

# ------------------------------------------------------
# GEMINI CHATBOT CONFIG
# ------------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini = genai.GenerativeModel("gemini-2.0-flash")

def chat_response(message: str) -> str:
    try:
        res = gemini.generate_content(message)
        return res.text
    except Exception as e:
        return f"Sorry, I had an issue answering that: {e}"

# ------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {"role": "user"/"bot", "msg": str}

# ------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------
st.sidebar.markdown("## 🧭 Navigate")
page = st.sidebar.radio(
    "",
    ["🏠 Home", "🐶 Breed Detector", "📜 Prediction History", "🤖 Chatbot"],
    label_visibility="collapsed"
)

# ------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------
if page == "🏠 Home":
    st.markdown(
        '<div class="paw-title"><span class="paw-icon">🐾</span>'
        '<span>Pawdentify – Dog Breed Detection System</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="card" style="margin-top:1.3rem;">
            <p style="color:{SUBTEXT_COLOR}; margin-bottom:0.9rem;">
            Welcome to <b>Pawdentify</b> – an AI-powered assistant that can identify dog breeds from images,
            explain breed characteristics, and answer your dog-related questions.</p>
            <ul style="color:{SUBTEXT_COLOR};">
                <li>🧠 Deep learning model trained on 120 dog breeds</li>
                <li>📸 Real-time dog breed prediction from your photos</li>
                <li>📘 Detailed breed information loaded from curated JSON</li>
                <li>🤖 Gemini-powered chatbot for all your dog questions</li>
                <li>📝 Prediction history with a visual gallery</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------
# BREED DETECTOR PAGE
# ------------------------------------------------------
elif page == "🐶 Breed Detector":
    st.markdown(
        '<div class="paw-title"><span class="paw-icon">🐕</span>'
        '<span>Dog Breed Detector</span></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a clear image of a dog", type=["jpg", "jpeg", "png", "webp"])

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(img, caption="Uploaded image", use_column_width=True)

        with col2:
            breed, conf = predict_breed(img)
            st.markdown(f"### 🐶 Predicted Breed: **{breed}**")
            st.markdown(f"#### 📊 Confidence: **{conf:.2f}%**")

            st.session_state.history.append({"image": img, "breed": breed, "conf": conf})

            if st.button("Know More 🐾"):
                st.markdown(get_breed_details(breed))

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------
# PREDICTION HISTORY PAGE (Gallery style)
# ------------------------------------------------------
elif page == "📜 Prediction History":
    st.markdown(
        '<div class="paw-title"><span class="paw-icon">📜</span>'
        '<span>Prediction History</span></div>',
        unsafe_allow_html=True
    )

    if len(st.session_state.history) == 0:
        st.info("No predictions yet. Try uploading a dog image first!")
    else:
        cols = st.columns(3)
        for idx, h in enumerate(st.session_state.history):
            with cols[idx % 3]:
                st.markdown('<div class="history-card">', unsafe_allow_html=True)
                st.image(h["image"], use_column_width=True)
                st.markdown(
                    f"<div style='margin-top:0.4rem; font-size:0.9rem;'>"
                    f"<b>{h['breed']}</b><br>"
                    f"<span class='small-tag'>Conf: {h['conf']:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                if st.button("Details", key=f"history_details_{idx}"):
                    st.markdown(get_breed_details(h["breed"]))
                st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------
# CHATBOT PAGE – ChatGPT-style
# ------------------------------------------------------
elif page == "🤖 Chatbot":
    st.markdown(
        '<div class="paw-title"><span class="paw-icon">🤖</span>'
        '<span>PawPal – Dog Chatbot</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"<p style='color:{SUBTEXT_COLOR}; margin-top:0.3rem;'>"
        "Ask anything about breeds, training, health (not a medical professional), grooming, or fun facts."
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="chat-window">', unsafe_allow_html=True)

    # Show chat history as bubbles
    for chat in st.session_state.chat:
        role = chat["role"]
        msg = chat["msg"]
        css_class = "chat-user" if role == "user" else "chat-bot"
        st.markdown(
            f'<div class="chat-msg {css_class}">{msg}</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)  # close chat-window

    # Chat input bar
    st.markdown('<div class="chat-input-bar">', unsafe_allow_html=True)
    col_inp, col_btn = st.columns([4, 1])

    with col_inp:
        user_msg = st.text_input("Type your message and press Send:", key="chat_input")

    with col_btn:
        if st.button("Send", key="chat_send"):
            if user_msg.strip():
                st.session_state.chat.append({"role": "user", "msg": user_msg})
                reply = chat_response(user_msg)
                st.session_state.chat.append({"role": "bot", "msg": reply})
                st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # close chat-input-bar
    st.markdown('</div>', unsafe_allow_html=True)  # close chat-wrapper
