import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "src" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "src" / "vectorizer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


class TweetRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Twitter Sentiment Analysis API is running"}

@app.post("/predict")
def predict_sentiment(request: TweetRequest):
    text_vector = vectorizer.transform([request.text])
    prediction = model.predict(text_vector)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    return {"sentiment": sentiment}


