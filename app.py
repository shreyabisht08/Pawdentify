import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
import google.generativeai as genai


# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="Dog Breed Detector",
    layout="wide",
)


# ------------------------------------------------------
# DARK/LIGHT THEME SWITCH
# ------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
    st.rerun()

theme_button = "🌙 Dark Mode" if st.session_state.theme == "light" else "☀️ Light Mode"
st.sidebar.button(theme_button, on_click=toggle_theme)

# Dynamic theme colors
BACKGROUND = "#ffffff" if st.session_state.theme == "light" else "#0e1117"
TEXT_COLOR = "#000000" if st.session_state.theme == "light" else "#fafafa"

st.markdown(
    f"""
    <style>
    body {{
        background-color: {BACKGROUND};
        color: {TEXT_COLOR};
    }}
    .stApp {{
        background-color: {BACKGROUND};
        color: {TEXT_COLOR};
    }}
    </style>
    """,
    unsafe_allow_html=True
)


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
    return {item["Breed"].strip(): item for item in data}

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
    breed = label_map[idx].strip()
    conf = float(np.max(pred) * 100)
    return breed, conf


# ------------------------------------------------------
# BREED DETAILS
# ------------------------------------------------------
def get_breed_details(breed):
    breed = breed.strip()
    if breed in breed_info:
        d = breed_info[breed]
        output = f"## 🐾 About {breed}\n"
        for k, v in d.items():
            if k != "Breed":
                output += f"**{k}:** {v}\n\n"
        return output
    return "❌ No details available."


# ------------------------------------------------------
# CHATBOT PAGE LOGIC
# ------------------------------------------------------
def chatbot_page():
    st.title("🤖 Dog AI Chatbot")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user_msg = st.text_input("Ask anything about dogs:", key="chat_input")

    if st.button("Send"):
        if user_msg.strip():
            st.session_state.chat.append({"role": "user", "msg": user_msg})

            reply = gemini.generate_content(user_msg).text
            st.session_state.chat.append({"role": "bot", "msg": reply})

            st.rerun()

    # Chat history display
    for m in st.session_state.chat:
        if m["role"] == "user":
            st.markdown(f"🧑 **You:** {m['msg']}")
        else:
            st.markdown(f"🤖 **Bot:** {m['msg']}")


# ------------------------------------------------------
# SIDEBAR NAV
# ------------------------------------------------------
page = st.sidebar.radio("Navigate", ["🏠 Home", "🐶 Breed Detector", "📜 Prediction History", "💬 Chatbot"])


# ------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------
if page == "🏠 Home":
    st.title("🐾 Dog Breed Detection System")

    st.write("""
    ### What this app does:
    - 🐶 Detect breed of any dog from an uploaded image  
    - 📘 Know detailed information about each breed  
    - 📝 View prediction history  
    - 🤖 Chat with an AI assistant about dogs  
    """)

    st.image("https://images.unsplash.com/photo-1517849845537-4d257902454a", use_column_width=True)


# ------------------------------------------------------
# BREED DETECTOR PAGE
# ------------------------------------------------------
elif page == "🐶 Breed Detector":
    st.title("🐶 Dog Breed Detector")

    uploaded = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=350)

        breed, conf = predict_breed(img)
        st.success(f"### 🐾 Breed: **{breed}**")
        st.info(f"### 📊 Confidence: **{conf:.2f}%**")

        # Save history
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({"image": img, "breed": breed, "conf": conf})

        if st.button("Know More 🐾"):
            st.markdown(get_breed_details(breed))


# ------------------------------------------------------
# HISTORY PAGE
# ------------------------------------------------------
elif page == "📜 Prediction History":
    st.title("📜 Prediction History")

    if "history" not in st.session_state or len(st.session_state.history) == 0:
        st.info("No predictions yet.")
    else:
        for h in st.session_state.history:
            st.image(h["image"], width=200)
            st.write(f"**Breed:** {h['breed']}")
            st.write(f"**Confidence:** {h['conf']:.2f}%")
            st.markdown("---")


# ------------------------------------------------------
# CHATBOT PAGE
# ------------------------------------------------------
elif page == "💬 Chatbot":
    chatbot_page()
