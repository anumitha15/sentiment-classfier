import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Sample dataset (you can expand later)
data = {
    "text": [
        "I am very happy",
        "I love this",
        "I am sad",
        "I hate this",
        "Feeling great",
        "Very bad experience"
    ],
    "label": [
        "Positive",
        "Positive",
        "Negative",
        "Negative",
        "Positive",
        "Negative"
    ]
}

df = pd.DataFrame(data)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(df["text"], df["label"])

joblib.dump(model, "model.pkl")

print("Model trained and saved!")