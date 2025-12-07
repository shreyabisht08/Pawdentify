import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Dog Chatbot", layout="wide")

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

st.title("🐾 Dog Chatbot")
st.write("Ask anything about dogs!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"🧑 **You:** {msg['text']}")
    else:
        st.markdown(f"🤖 **AI:** {msg['text']}")

query = st.text_input("Type your message:")

if st.button("Send"):
    if query.strip():
        st.session_state.chat_history.append({"role": "user", "text": query})
        reply = model.generate_content(query).text
        st.session_state.chat_history.append({"role": "bot", "text": reply})
        st.experimental_rerun()
