import streamlit as st

from chatbot import FAQChatbot


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 FAQ Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions and get answers from our FAQ knowledge base.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD CHATBOT
# ============================================================

@st.cache_resource
def load_chatbot():

    return FAQChatbot(
        faq_file="data/faqs.csv",
        threshold=0.20
    )


try:

    chatbot = load_chatbot()

except Exception as e:

    st.error(
        f"❌ Error loading chatbot: {e}"
    )

    st.stop()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

user_question = st.chat_input(
    "💬 Ask your question..."
)


if user_question:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):

        st.markdown(user_question)


    # --------------------------------------------------------
    # Get chatbot response
    # --------------------------------------------------------

    result = chatbot.get_response(
        user_question
    )

    answer = result["answer"]
    score = result["score"]
    matched_question = result["matched_question"]


    # --------------------------------------------------------
    # Display chatbot response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):

        st.markdown(answer)

        # Show matching information
        if matched_question:

            with st.expander(
                "🔍 View Matching Information"
            ):

                st.write(
                    f"**Matched FAQ:** {matched_question}"
                )

                st.write(
                    f"**Similarity Score:** {score:.2f}"
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 FAQ Chatbot")

    st.markdown(
        """
        ### 📚 About

        This chatbot uses:

        🧠 **NLTK**

        📊 **TF-IDF**

        📐 **Cosine Similarity**

        🐍 **Python**

        🎨 **Streamlit**
        """
    )

    st.divider()

    st.markdown(
        """
        ### 💡 Try asking

        • How can I track my order?

        • Can I cancel my order?

        • What payment methods do you accept?

        • How long does delivery take?

        • How can I reset my password?
        """
    )