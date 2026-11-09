
import json
import os
import inspect
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.preprocessing import preprocess


# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "intents.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.pkl")

LANGUAGES = ["fr", "en"]


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    dataset = {"fr": [], "en": []}

    for intent in data["intents"]:
        tag = intent["tag"]

        for pattern in intent["patterns"]:
            text = pattern["text"]
            language = pattern["language"]

            if language in LANGUAGES:
                tokens = preprocess(text, language)

                if tokens:
                    dataset[language].append((" ".join(tokens), tag))

    return dataset


def create_calibrated_svm():
    # Support different scikit-learn versions
    parameters = inspect.signature(CalibratedClassifierCV).parameters

    if "estimator" in parameters:
        return CalibratedClassifierCV(
            estimator=LinearSVC(),
            method="sigmoid",
            cv=3
        )

    return CalibratedClassifierCV(
        base_estimator=LinearSVC(),
        method="sigmoid",
        cv=3
    )


def train_model(dataset):
    models = {}

    for language in LANGUAGES:
        print(f"\nTraining language: {language.upper()}")

        samples = dataset[language]

        X = [sample[0] for sample in samples]
        y = [sample[1] for sample in samples]

        if not X:
            print(f"No data found for {language}.")
            continue

        if language == "fr":
            classifier = create_calibrated_svm()
            print("Selected algorithm: Calibrated Linear SVM")
        else:
            classifier = LogisticRegression(max_iter=1000)
            print("Selected algorithm: Logistic Regression")

        model = Pipeline([
            ("vectorizer", CountVectorizer(lowercase=False)),
            ("classifier", classifier)
        ])

        model.fit(X, y)

        models[language] = model

        print(f"{language.upper()} model trained successfully!")
        print("Training examples:", len(X))
        print("Number of intents:", len(set(y)))

    return models


if __name__ == "__main__":
    print("Loading chatbot dataset...")

    dataset = load_data()

    print("Dataset loaded successfully!")

    models = train_model(dataset)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(models, MODEL_PATH)

    print("\nTraining completed!")
    print("Models saved at:", MODEL_PATH)
