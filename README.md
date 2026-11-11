
# Bilingual NLP Chatbot (French & English)

A bilingual chatbot developed with Python, NLTK and Machine Learning. It identifies user intentions in French and English and generates automatic responses through a Streamlit web interface.

## Project Overview

This project demonstrates how Natural Language Processing (NLP) and supervised Machine Learning can be used to build an intent-based chatbot.

The chatbot supports six intentions:

- Greeting
- Goodbye
- Thanks
- Help
- Name
- Hours

It includes text preprocessing, intent classification, confidence-based prediction and automatic response generation.

## Technologies

- Python
- NLTK — text preprocessing
- Scikit-learn — Machine Learning
- Joblib — model persistence
- Streamlit — web interface

## Machine Learning Algorithms

Different algorithms were evaluated for each language.

| Language | Selected algorithm |
|---|---|
| French | Calibrated Linear SVM |
| English | Logistic Regression |

The French Linear SVM is calibrated to provide probability estimates used by the chatbot's confidence threshold.

## Model Evaluation

The models were evaluated using 3-fold stratified cross-validation.

| Language | Algorithm | Accuracy | Macro F1-score |
|---|---|---:|---:|
| French | Naive Bayes | 76.22% | 76.28% |
| French | Logistic Regression | 75.34% | 75.27% |
| French | Linear SVM | 77.91% | 77.52% |
| English | Naive Bayes | 63.75% | 61.43% |
| English | Logistic Regression | 66.11% | 65.29% |
| English | Linear SVM | 62.18% | 61.13% |

The final models were selected based on macro F1-score.

## Project Structure

```text
nlp-chatbot/
│
├── app.py
├── data/
│   └── intents.json
│
├── models/
│   └── chatbot_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── chatbot.py
│   └── evaluate.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
https://github.com/KATXENNN/bilingual-nlp-chatbot
cd nlp-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download NLTK resources

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## Run the Application

### Streamlit Web Interface

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

### Command-Line Chatbot

```bash
python -m src.chatbot
```

## Train the Models Again

If you modify the dataset or preprocessing:

```bash
python -m src.train
```

The trained models will be saved in the `models/` directory.

## Main Features

- French and English language support.
- Text preprocessing using NLTK.
- Intent classification using Machine Learning.
- Automatic response generation.
- Confidence-based prediction.
- Rejection of low-confidence questions.
- Interactive Streamlit interface.
- Conversation history.

## Limitations

This is an intent-based chatbot, not a general-purpose conversational AI system.

It can respond to patterns associated with its six trained intentions. Questions outside these categories may be rejected or incorrectly classified.

Confidence thresholds are language-specific and should be further evaluated using a dedicated set of known and out-of-scope test examples.

## Academic Project

Developed as part of a Master's-level project in Natural Language Processing (NLP).

**Main objective:** Apply NLP techniques and supervised Machine Learning algorithms to build a bilingual automatic-response chatbot.
