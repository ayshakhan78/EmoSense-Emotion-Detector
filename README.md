# EmoSense — Emotion Detection from Text

A machine learning web application that detects emotions from text using Natural Language Processing (NLP). Built with Python, scikit-learn, and deployed with Streamlit.

## Live Demo
https://emosense-emotion-detector-wxqfk7ecm8vrna2avbeepq.streamlit.app/

##  About the Project

EmoSense analyzes any piece of text and classifies it into one of 6 human emotions in real time. The model was trained on a travel reviews dataset and uses classical NLP techniques to understand the emotional tone behind words.

This project covers the full ML pipeline — from raw data to a deployed, interactive web application.

##  Emotions Detected

| Emotion | Emoji |
|---------|-------|
| Joy | 😄 |
| Sadness | 😢 |
| Anger | 😠 |
| Fear | 😨 |
| Love | ❤️ |
| Surprise | 😲 |


##  Model Performance

| Model | Vectorizer | Accuracy |
|-------|-----------|----------|
| Logistic Regression | Bag of Words | **88%** ✅ |
| Logistic Regression | TF-IDF | 86% |
| Naive Bayes | Bag of Words | ~82% |
| SVM | Bag of Words | ~84% |

> Best model selected: **Logistic Regression + Bag of Words (88%)**

## Tech Stack

- **Language:** Python
- **ML Library:** scikit-learn
- **NLP:** CountVectorizer (Bag of Words)
- **Web App:** Streamlit
- **Model Saving:** joblib
- **Development:** Google Colab, VS Code

## How It Works

1. User types any text into the input box
2. Text is transformed into a numerical vector using the trained Bag of Words vectorizer
3. The Logistic Regression model predicts the emotion class
4. The app displays the detected emotion with confidence score and probability breakdown for all 6 emotions
