import streamlit as st
import google.generativeai as genai

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="Dog Chatbot", layout="wide")

# ------------------------------------------------------
# GEMINI INIT
# ------------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

st.title("🐾 Dog Chatbot – Ask Anything About Dogs")

st.write("Chat with your AI dog assistant. Ask about breeds, care, training, behavior, etc.")

# ------------------------------------------------------
# CHAT HISTORY
# ------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------------------------------
# DISPLAY CHAT
# ------------------------------------------------------
for chat in st.session_state.chat_history:
    role = chat["role"]
    text = chat["text"]

    if role == "user":
        st.markdown(
            f"""
            <div style="background:#8b5e34;color:white;padding:10px;border-radius:10px;margin:6px;text-align:right;">
                {text}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background:#f0e7d8;padding:10px;border-radius:10px;margin:6px;text-align:left;">
                {text}
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------------------------------------------
# USER INPUT
# ------------------------------------------------------
query = st.text_input("Type your message...", key="chat_input")

if st.button("Send"):
    if query.strip():
        st.session_state.chat_history.append({"role": "user", "text": query})

        response = model.generate_content(query)
        reply = response.text

        st.session_state.chat_history.append({"role": "bot", "text": reply})

        st.experimental_rerun()
