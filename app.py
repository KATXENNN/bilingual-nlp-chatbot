
import json
import os
import random
import joblib
import streamlit as st

from src.preprocessing import preprocess


# =====================================
# 1. PROJECT CONFIGURATION
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "chatbot_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "intents.json")

CONFIDENCE_THRESHOLDS = {
    "fr": 0.50,
    "en": 0.40
}


# =====================================
# 2. STREAMLIT PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="NLP Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
    .main {
        max-width: 900px;
        margin: auto;
    }

    .stChatMessage {
        border-radius: 12px;
    }

    h1 {
        text-align: center;
    }

    .subtitle {
        text-align: center;
        color: #888888;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================
# 3. LOAD MODELS AND DATA
# =====================================

@st.cache_resource
def load_resources():

    models = joblib.load(MODEL_PATH)

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return models, data


models, data = load_resources()


# =====================================
# 4. CHATBOT FUNCTIONS
# =====================================

def get_response(tag, language):

    for intent in data["intents"]:

        if intent["tag"] == tag:

            responses = [
                item["text"]
                for item in intent["responses"]
                if item["language"] == language
            ]

            if responses:
                return random.choice(responses)

    return (
        "Je ne comprends pas votre demande."
        if language == "fr"
        else "I don't understand your request."
    )


def generate_response(user_input, language):

    tokens = preprocess(user_input, language)

    if not tokens:
        return {
            "response": (
                "Veuillez saisir une phrase compréhensible."
                if language == "fr"
                else "Please enter a meaningful sentence."
            ),
            "accepted": False
        }

    processed_text = " ".join(tokens)

    model = models.get(language)

    if model is None:
        return {
            "response": "Modèle indisponible." if language == "fr"
            else "Model not available.",
            "accepted": False
        }

    probabilities = model.predict_proba([processed_text])[0]

    confidence = max(probabilities)

    predicted_intent = model.classes_[probabilities.argmax()]

    if confidence < CONFIDENCE_THRESHOLDS[language]:

        return {
            "response": (
                "Désolé, je n'ai pas bien compris."
                if language == "fr"
                else "Sorry, I didn't understand."
            ),
            "accepted": False
        }

    return {
        "response": get_response(predicted_intent, language),
        "accepted": True,
        "intent": predicted_intent,
        "confidence": confidence
    }


# =====================================
# 5. SESSION STATE
# =====================================

def reset_chat():
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================
# 6. SIDEBAR
# =====================================

st.sidebar.title("⚙️ Settings")

language = st.sidebar.selectbox(
    "Choose your language / Choisir la langue",
    options=["en", "fr"],
    format_func=lambda x: "English" if x == "en" else "Français",
    key="language",
    on_change=reset_chat
)

if language == "fr":
    st.sidebar.markdown("### À propos")
    st.sidebar.write("Chatbot bilingue développé avec Python, NLTK et Machine Learning.")
    st.sidebar.write("**Algorithmes :**")
    st.sidebar.write("- Linear SVM (Français)")
    st.sidebar.write("- Logistic Regression (English)")
    clear_label = "Effacer la conversation"
else:
    st.sidebar.markdown("### About")
    st.sidebar.write("Bilingual chatbot developed using Python, NLTK and Machine Learning.")
    st.sidebar.write("**Algorithms:**")
    st.sidebar.write("- Linear SVM (French)")
    st.sidebar.write("- Logistic Regression (English)")
    clear_label = "Clear conversation"

if st.sidebar.button(clear_label, use_container_width=True):
    reset_chat()
    st.rerun()


# =====================================
# 7. MAIN INTERFACE
# =====================================

st.title("🤖 NLP Chatbot")

if language == "fr":
    st.markdown(
        '<p class="subtitle">Votre assistant intelligent bilingue</p>',
        unsafe_allow_html=True
    )
    welcome_message = "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
    input_placeholder = "Écrivez votre message..."
else:
    st.markdown(
        '<p class="subtitle">Your intelligent bilingual assistant</p>',
        unsafe_allow_html=True
    )
    welcome_message = "Hello! How can I help you today?"
    input_placeholder = "Type your message..."


# Display welcome message
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write(welcome_message)


# Display conversation history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message["role"] == "assistant" and message.get("accepted"):

            st.caption(
                f"Intent: {message['intent']} | "
                f"Confidence: {message['confidence']:.2%}"
            )


# =====================================
# 8. CHAT INPUT AND RESPONSE
# =====================================

user_input = st.chat_input(input_placeholder)

if user_input:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Generate chatbot response
    result = generate_response(user_input, language)

    assistant_message = {
        "role": "assistant",
        "content": result["response"],
        "accepted": result["accepted"]
    }

    if result["accepted"]:
        assistant_message["intent"] = result["intent"]
        assistant_message["confidence"] = result["confidence"]

    st.session_state.messages.append(assistant_message)

    st.rerun()
