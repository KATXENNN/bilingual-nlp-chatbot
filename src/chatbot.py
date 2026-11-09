
import json
import os
import random
import joblib

from src.preprocessing import preprocess


# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "chatbot_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "intents.json")


# Confidence thresholds by language
CONFIDENCE_THRESHOLDS = {
    "fr": 0.50,
    "en": 0.40
}


# Load trained models
models = joblib.load(MODEL_PATH)


# Load chatbot responses
with open(DATA_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)


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


def chatbot():

    print("\n===== NLP CHATBOT =====")
    print("1. Français")
    print("2. English")
    print("0. Quit")

    choice = input("\nChoose your language: ")

    if choice == "0":
        return

    if choice not in ("1", "2"):
        print("Invalid choice.")
        return chatbot()

    language = "fr" if choice == "1" else "en"

    print("\nChatbot started! Type 'exit' to stop.")

    while True:

        user_input = input("\nYou: ")

        if user_input.lower().strip() in ("exit", "quit"):
            print("Chatbot: Goodbye!")
            break

        # Preprocessing
        tokens = preprocess(user_input, language)

        if not tokens:
            print("Chatbot: Please enter a meaningful sentence.")
            continue

        processed_text = " ".join(tokens)

        # Intent prediction
        model = models.get(language)

        if model is None:
            print("Chatbot: Model not available.")
            continue

        probabilities = model.predict_proba([processed_text])[0]

        confidence = max(probabilities)

        predicted_intent = model.classes_[probabilities.argmax()]

        # Confidence threshold
        if confidence < CONFIDENCE_THRESHOLDS[language]:

            response = (
                "Désolé, je n'ai pas bien compris."
                if language == "fr"
                else "Sorry, I didn't understand."
            )

            print("Chatbot:", response)

        else:

            response = get_response(predicted_intent, language)

            print("Chatbot:", response)
            print("Detected intent:", predicted_intent)
            print("Confidence:", round(confidence * 100, 2), "%")


if __name__ == "__main__":
    chatbot()
