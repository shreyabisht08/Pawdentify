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
    page_title="🐾 Pawdentify - Dog Breed Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize theme in session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Dynamic CSS styling based on theme
def get_theme_colors():
    if st.session_state.theme == "dark":
        return {
            "bg_primary": "#0e1117",
            "bg_secondary": "#161b22",
            "text_primary": "#e6edf3",
            "text_secondary": "#8b949e",
            "card_bg": "#161b22",
            "border_color": "#30363d",
            "accent": "#667eea",
            "button_bg": "#667eea",
            "button_hover_bg": "#764ba2",
            "info_bg": "#0d3a66",
            "success_bg": "#033a16",
        }
    else:
        return {
            "bg_primary": "#ffffff",
            "bg_secondary": "#f6f8fb",
            "text_primary": "#000000",
            "text_secondary": "#666666",
            "card_bg": "#ffffff",
            "border_color": "#e0e0e0",
            "accent": "#667eea",
            "button_bg": "#667eea",
            "button_hover_bg": "#764ba2",
            "info_bg": "#e3f2fd",
            "success_bg": "#e8f5e9",
        }

colors = get_theme_colors()

# Apply dynamic CSS styling
st.markdown(f"""
    <style>
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    body, .stApp {{
        background-color: {colors['bg_primary']};
        color: {colors['text_primary']};
    }}
    
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    
    .main-header {{
        text-align: center;
        padding: 30px 0;
        background: linear-gradient(135deg, {colors['accent']} 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    
    .main-header h1 {{
        font-size: 2.5em;
        margin-bottom: 10px;
        font-weight: 700;
    }}
    
    .main-header p {{
        font-size: 1.1em;
        opacity: 0.95;
    }}
    
    .card {{
        background: {colors['card_bg']};
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid {colors['accent']};
        transition: transform 0.2s, box-shadow 0.2s;
        color: {colors['text_primary']};
    }}
    
    .card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    .breed-card {{
        background: {'linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%)' if st.session_state.theme == 'dark' else 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)'};
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: {colors['text_primary']};
    }}
    
    .breed-details {{
        background: {colors['card_bg']};
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        border-left: 5px solid {colors['accent']};
        color: {colors['text_primary']};
    }}
    
    .breed-details h2 {{
        color: {colors['accent']};
        margin-bottom: 15px;
    }}
    
    .breed-details-item {{
        margin: 10px 0;
        padding: 10px;
        background: {colors['bg_secondary']};
        border-radius: 5px;
        color: {colors['text_primary']};
    }}
    
    .info-box {{
        background: {colors['info_bg']};
        border-left: 4px solid {colors['accent']};
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        color: {colors['text_primary']};
    }}
    
    .success-box {{
        background: {colors['success_bg']};
        border-left: 4px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        color: {colors['text_primary']};
    }}
    
    .sidebar-nav {{
        padding: 20px 0;
    }}
    
    .sidebar-nav-item {{
        padding: 12px 15px;
        margin: 8px 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s;
    }}
    
    .stButton > button {{
        width: 100%;
        padding: 12px 24px;
        background: linear-gradient(135deg, {colors['button_bg']} 0%, {colors['button_hover_bg']} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1em;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }}
    
    [data-testid="stMetricValue"] {{
        color: {colors['accent']};
        font-weight: 700;
    }}
    </style>
""", unsafe_allow_html=True)


# DARK/LIGHT THEME SWITCH
def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
    st.rerun()

theme_button = "🌙 Dark" if st.session_state.theme == "light" else "☀️ Light"
with st.sidebar:
    st.button(theme_button, on_click=toggle_theme, use_container_width=True)


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
    # Extract just the breed names (remove ImageNet ID prefix like 'n02085620-')
    label_dict = {}
    for k, v in classes.items():
        # Remove the ImageNet ID prefix (everything before the dash)
        breed_name = k.split('-', 1)[1] if '-' in k else k
        label_dict[int(v)] = breed_name
    return label_dict

label_map = load_labels()


@st.cache_data
def load_info():
    with open("120_breeds_new.json", "r") as f:
        data = json.load(f)
    # Create mappings for different name formats
    breed_dict = {}
    for item in data:
        breed_name = item["Breed"].strip()
        # Add original breed name
        breed_dict[breed_name] = item
        # Add alternative formats for matching
        breed_dict[breed_name.lower()] = item
        breed_dict[breed_name.replace("_", " ")] = item
        breed_dict[breed_name.replace("_", " ").lower()] = item
    return breed_dict

breed_info = load_info()


# Create a mapping from model labels to breed info
def normalize_breed_name(breed):
    """Normalize breed name for matching with JSON data"""
    breed = breed.strip()
    
    # First, try to find exact match (case-insensitive)
    for key in breed_info.keys():
        if key.lower() == breed.lower():
            return key
    
    # Try with underscores/spaces normalization
    breed_normalized = breed.lower().replace(" ", "_").replace("-", "_")
    for key in breed_info.keys():
        key_normalized = key.lower().replace(" ", "_").replace("-", "_")
        if key_normalized == breed_normalized:
            return key
    
    # If no match found, return original (will return None in get_breed_details)
    return None


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
# BREED DETAILS - ENHANCED
# ------------------------------------------------------
def get_breed_details(breed):
    normalized_breed = normalize_breed_name(breed)
    if normalized_breed and normalized_breed in breed_info:
        return breed_info[normalized_breed]
    return None


# ------------------------------------------------------
# CHATBOT PAGE LOGIC
# ------------------------------------------------------
def chatbot_page():
    st.title("🤖 Dog AI Chatbot")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Display chat history with better styling
    chat_container = st.container()
    with chat_container:
        for m in st.session_state.chat:
            if m["role"] == "user":
                st.markdown(f"""
                    <div style='text-align: right; margin: 15px 0;'>
                        <span style='background: #667eea; color: white; padding: 10px 15px; border-radius: 10px; display: inline-block;'>
                        👤 {m['msg']}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='text-align: left; margin: 15px 0;'>
                        <span style='background: #e3f2fd; color: #333; padding: 10px 15px; border-radius: 10px; display: inline-block; border-left: 3px solid #667eea;'>
                        🤖 {m['msg']}
                        </span>
                    </div>
                """, unsafe_allow_html=True)

    # Input section
    st.markdown("---")
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        user_msg = st.text_input("Ask anything about dogs:", key="chat_input", label_visibility="collapsed")
    with col2:
        send_clicked = st.button("Send", use_container_width=True)

    if send_clicked and user_msg.strip():
        st.session_state.chat.append({"role": "user", "msg": user_msg})
        try:
            reply = gemini.generate_content(user_msg).text
            st.session_state.chat.append({"role": "bot", "msg": reply})
        except Exception as e:
            st.error(f"Error: {str(e)}")
        st.rerun()


# ------------------------------------------------------
# SIDEBAR NAV
# ------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Select Page", ["🏠 Home", "🐶 Breed Detector", "📜 History", "💬 Chatbot"], label_visibility="collapsed")


# ------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------
if page == "🏠 Home":
    st.markdown("""
        <div class='main-header'>
            <h1>🐾 Pawdentify</h1>
            <p>Intelligent Dog Breed Detection & AI Assistant</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3>🐶 Features</h3>
            <ul style='list-style: none; padding: 0;'>
                <li>✅ Detect dog breed from image with high accuracy</li>
                <li>✅ View detailed breed information</li>
                <li>✅ Track prediction history</li>
                <li>✅ AI-powered chatbot for dog experts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3>⚡ How to Use</h3>
            <ol style='padding-left: 20px;'>
                <li>Go to <b>Breed Detector</b></li>
                <li>Upload a dog image</li>
                <li>Click <b>Know More</b> for details</li>
                <li>Ask our AI chatbot any questions</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Display sample image
    try:
        st.image("https://images.unsplash.com/photo-1633722715463-d30628519e73?w=800", use_column_width=True)
    except:
        st.info("📸 Featuring: A beautiful dog breed showcase")


# ------------------------------------------------------
# BREED DETECTOR PAGE
# ------------------------------------------------------
elif page == "🐶 Breed Detector":
    st.markdown("""
        <div class='main-header'>
            <h1>🐶 Breed Detector</h1>
            <p>Upload an image to identify the dog breed</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([0.5, 0.5])
    
    with col1:
        uploaded = st.file_uploader("📤 Upload a dog image", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded:
        with col2:
            st.write("")
            st.write("")
            st.write("**Preview:**")
        
        col1, col2 = st.columns([0.5, 0.5])
        
        with col1:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_column_width=True, caption="Uploaded Image")
        
        with col2:
            st.write("")
            with st.spinner("🔍 Analyzing image..."):
                breed, conf = predict_breed(img)
            
            # Display results in styled boxes
            st.markdown(f"""
                <div class='success-box'>
                    <h3>🎯 Detected Breed</h3>
                    <h2 style='color: #667eea; margin: 10px 0;'>{breed}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class='info-box'>
                    <h4>📊 Confidence Score</h4>
                    <h3 style='color: #667eea;'>{conf:.2f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Save to history
            if "history" not in st.session_state:
                st.session_state.history = []
            
            st.session_state.history.append({"image": img, "breed": breed, "conf": conf})
        
        # Know More Section
        st.markdown("---")
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button("📖 Know More About This Breed", use_container_width=True):
                st.session_state.show_details = not st.session_state.get("show_details", False)
        with col2:
            if st.session_state.get("show_details", False):
                if st.button("Close ✕", use_container_width=True):
                    st.session_state.show_details = False
                    st.rerun()
        
        if st.session_state.get("show_details", False):
            breed_details = get_breed_details(breed)
            if breed_details:
                st.markdown(f"""
                    <div class='breed-details'>
                        <h2>🐾 About {breed}</h2>
                """, unsafe_allow_html=True)
                
                # Display each detail nicely
                for key, value in breed_details.items():
                    if key != "Breed":
                        st.markdown(f"""
                            <div class='breed-details-item'>
                                <b>{key}:</b> {value}
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("❌ No detailed information available for this breed.")


# ------------------------------------------------------
# HISTORY PAGE
# ------------------------------------------------------
elif page == "📜 History":
    st.markdown("""
        <div class='main-header'>
            <h1>📜 Prediction History</h1>
            <p>Your previous breed detections</p>
        </div>
    """, unsafe_allow_html=True)

    if "history" not in st.session_state or len(st.session_state.history) == 0:
        st.info("📭 No predictions yet. Start by detecting a breed!")
    else:
        for idx, h in enumerate(reversed(st.session_state.history)):
            with st.container():
                col1, col2 = st.columns([0.3, 0.7])
                with col1:
                    st.image(h["image"], use_column_width=True)
                with col2:
                    st.markdown(f"""
                        <div class='breed-card'>
                            <h3>🐶 {h['breed']}</h3>
                            <p><b>Confidence:</b> {h['conf']:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("---")


# ------------------------------------------------------
# CHATBOT PAGE
# ------------------------------------------------------
elif page == "💬 Chatbot":
    chatbot_page()
