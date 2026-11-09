
import os

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix

from src.train import load_data, LANGUAGES


# Algorithms to compare
CLASSIFIERS = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(max_iter=5000)
}

# Evaluation metrics
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}


def evaluate_models(dataset):

    for language in LANGUAGES:

        print("\n" + "=" * 55)
        print(f"MODEL EVALUATION: {language.upper()}")
        print("=" * 55)

        samples = dataset[language]

        X = [sample[0] for sample in samples]
        y = [sample[1] for sample in samples]

        class_counts = Counter(y)

        print("\nNumber of examples:", len(X))
        print("Number of intents:", len(class_counts))
        print("Intent distribution:", dict(class_counts))

        min_samples = min(class_counts.values())

        if min_samples < 2:
            print("Not enough examples for cross-validation.")
            continue

        n_splits = min(3, min_samples)

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        results = {}

        # Compare algorithms
        for name, classifier in CLASSIFIERS.items():

            model = Pipeline([
                ("vectorizer", CountVectorizer(lowercase=False)),
                ("classifier", classifier)
            ])

            scores = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=SCORING
            )

            results[name] = scores

            print(f"\n--- {name} ---")

            for metric in SCORING:

                values = scores[f"test_{metric}"]

                print(
                    f"{metric.capitalize()}: "
                    f"{values.mean() * 100:.2f}% "
                    f"(+/- {values.std() * 100:.2f})"
                )

        # Select the best algorithm using macro F1-score
        best_model_name = max(
            results,
            key=lambda name: results[name]["test_f1"].mean()
        )

        print("\nBest algorithm:", best_model_name)

        # Generate cross-validated predictions
        best_model = Pipeline([
            ("vectorizer", CountVectorizer(lowercase=False)),
            ("classifier", CLASSIFIERS[best_model_name])
        ])

        predictions = cross_val_predict(
            best_model,
            X,
            y,
            cv=cv
        )

        # Confusion matrix
        labels = sorted(class_counts.keys())

        matrix = confusion_matrix(
            y,
            predictions,
            labels=labels
        )

        print("\nConfusion Matrix:")
        print("Labels:", labels)
        print(matrix)


if __name__ == "__main__":

    print("Loading dataset for evaluation...")

    dataset = load_data()

    print("Dataset loaded successfully!")

    evaluate_models(dataset)

    print("\nEvaluation completed!")
