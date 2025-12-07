import streamlit as st
import google.generativeai as genai

st.set_page_config(
    page_title="🐾 Dog Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling for ChatGPT-like interface
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 30px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5em;
        margin-bottom: 10px;
        font-weight: 700;
    }
    
    .chat-message-user {
        display: flex;
        justify-content: flex-end;
        margin: 15px 0;
    }
    
    .chat-message-bot {
        display: flex;
        justify-content: flex-start;
        margin: 15px 0;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 15px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .bot-bubble {
        background: #f0f0f0;
        color: #333;
        padding: 12px 16px;
        border-radius: 15px;
        max-width: 70%;
        word-wrap: break-word;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .chat-container {
        height: 600px;
        overflow-y: auto;
        padding: 20px;
        background: white;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    .input-container {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-top: 1px solid #e0e0e0;
    }
    
    .stButton > button {
        width: 100%;
        padding: 12px 24px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1em;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

st.markdown("""
    <div class='main-header'>
        <h1>🐾 Dog AI Chatbot</h1>
        <p>Ask anything about dogs, breeds, care, and more!</p>
    </div>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat display container
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if len(st.session_state.chat_history) == 0:
        st.markdown("""
            <div class='info-box'>
                <h3>👋 Welcome!</h3>
                <p>I'm your friendly dog expert AI. Ask me anything about dog breeds, health, training, behavior, or any other dog-related topics!</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                    <div class='chat-message-user'>
                        <div class='user-bubble'>
                            {msg['text']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='chat-message-bot'>
                        <div class='bot-bubble'>
                            {msg['text']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Input section
st.markdown('<div class="input-container">', unsafe_allow_html=True)

col1, col2 = st.columns([0.88, 0.12])

with col1:
    query = st.text_input(
        "Type your message...",
        placeholder="Ask me about dog breeds, care, training...",
        label_visibility="collapsed"
    )

with col2:
    send_btn = st.button("Send ✉️", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if send_btn:
    if query.strip():
        # Add user message
        st.session_state.chat_history.append({"role": "user", "text": query})
        
        # Get AI response
        try:
            with st.spinner("🤖 Thinking..."):
                reply = model.generate_content(query).text
                st.session_state.chat_history.append({"role": "bot", "text": reply})
        except Exception as e:
            st.error(f"Error: {str(e)}")
        
        st.rerun()
    else:
        st.warning("Please type a message first!")
