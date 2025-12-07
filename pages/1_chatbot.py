import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Dog Chatbot", layout="wide")

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

st.title("🐾 Dog Chatbot")
st.write("Ask anything about dogs!")

# -------------------------
# CHAT HISTORY
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display old chats
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"🧑 **You:** {chat['text']}")
    else:
        st.markdown(f"🤖 **AI:** {chat['text']}")

# Input
query = st.text_input("Type your message:")

if st.button("Send"):
    if query.strip():
        st.session_state.chat_history.append({"role": "user", "text": query})

        reply = model.generate_content(query).text
        st.session_state.chat_history.append({"role": "bot", "text": reply})

        st.experimental_rerun()
