import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer


# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
french_stemmer = SnowballStemmer("french")

# Load stopwords
STOPWORDS = {
    "fr": set(stopwords.words("french")),
    "en": set(stopwords.words("english"))
}

# Preserve negation words because they can change meaning
NEGATIONS = {
    "fr": {"ne", "pas", "plus", "jamais", "aucun", "rien"},
    "en": {
        "no", "not", "nor", "never", "none",
        "don't", "doesn't", "didn't",
        "can't", "cannot", "won't",
        "wouldn't", "shouldn't",
        "isn't", "aren't", "wasn't", "weren't",
        "haven't", "hasn't", "hadn't"
    }
}

# Preserve important question words
STOPWORDS["en"] -= {"what", "do", "how", "why", "where", "who"}

for language in STOPWORDS:
    STOPWORDS[language] -= NEGATIONS[language]


def tokenize(text):
    """
    Split text into words while preserving apostrophes.
    Supports French accents and English words.
    """
    return re.findall(r"\b[\w]+(?:['’][\w]+)?\b", text.lower())

def correct_common_typos(text):
    corrections = {
        "lsot": "lost",
        "helo": "hello",
        "helllo": "hello",
        "thnaks": "thanks",
        "thx": "thanks"
    }

    words = text.split()
    corrected_words = [
        corrections.get(word.lower(), word)
        for word in words
    ]

    return " ".join(corrected_words)

def preprocess(text, language="fr"):
    """
    Clean and normalize a sentence.

    Args:
        text (str): Input sentence.
        language (str): 'fr' or 'en'.

    Returns:
        list: Processed tokens.
    """

    if language not in ("fr", "en"):
        raise ValueError("Language must be 'fr' or 'en'.")

    text = correct_common_typos(text)
    tokens = tokenize(text)

    processed_tokens = []

    for token in tokens:

        # Remove stopwords
        if token in STOPWORDS[language]:
            continue

        # English lemmatization
        if language == "en":
            token = lemmatizer.lemmatize(token)

        # French stemming
        elif language == "fr":
            token = french_stemmer.stem(token)

        processed_tokens.append(token)

    return processed_tokens


if __name__ == "__main__":

    french_text = "Bonjour, j'ai besoin d'aide !"
    english_text = "Hello, I need some help!"

    print("French preprocessing:")
    print(preprocess(french_text, "fr"))

    print("\nEnglish preprocessing:")
    print(preprocess(english_text, "en"))