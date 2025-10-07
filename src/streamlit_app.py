import streamlit as st
from agent import invoke_graph
# --- PAGE CONFIG ---
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

st.title("📄 Chat with your PDF")

# --- SIDEBAR ---
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
if st.sidebar.button("🔄 Start New Chat"):
    st.session_state.chat_history = []
    st.session_state.user_query =""

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_query" not in st.session_state:
    st.session_state.user_query =""
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "count" not in st.session_state:
    st.session_state.count = 0

# --- CHAT INTERFACE ---
if uploaded_file:
    st.success("PDF processed! You can start chatting below 👇")
    chat_holder = st.container()

    # st.session_state.user_query=""
    st.markdown(
     """
    <style>
    div[data-testid="stTextInput"] {
        position: sticky;
        bottom: 10px;
        width: 100%;       /* full width of content area */
        max-width: 100%;
        background-color: black;
        color: black;
    }
    div[data-testid="stTextInputRootElement"] {
        border: 2px solid #CFCFCF;
        border-radius:10px
    }
    div[data-baseweb="base-input"] {
            background-color: #5E5E5E;
    }
    div[data-testid="stElementContainer"] {
        position:fixed,
        bottom:50px;
        width:-webkit-fill-available;
        right:20px;
        left:20px;
    }
    
    /* Placeholder color */
    div[data-testid="stTextInput"] input::placeholder {
        color: gray;
    }

    div[data-testid="stTextInput"] input {
        color: #CFCFCF;
    }

    /* Add padding at the bottom of the chat container */
    div[data-testid="stVerticalBlock"] {
        padding-bottom: 60px;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    user_query = st.text_input("Ask something about your PDF:", key="user_query")
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with chat_holder:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**🧑 You:** {msg['content']}")
                else:
                    st.markdown(f"**🤖 Bot:** {msg['content']}")
                st.markdown("---")
                st.session_state.count+=1
        with st.spinner("Thinking..."):
            file_bytes=uploaded_file.read()
            result = invoke_graph(file_bytes,user_query)


        st.session_state.chat_history.append({"role": "bot", "content": result})
        with chat_holder:
            for msg in st.session_state.chat_history[st.session_state.count::]:
                if msg["role"] == "user":
                    st.markdown(f"**🧑 You:** {msg['content']}")
                else:
                    st.markdown(f"**🤖 Bot:** {msg['content']}")
                st.markdown("---")
        st.session_state.count=0
else:
    st.info("👆 Upload a PDF to start chatting.")

